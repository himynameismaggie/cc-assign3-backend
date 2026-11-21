import json
import logging
import os
import re
from uuid import uuid4

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from opensearchpy import OpenSearch, RequestsAWSV4SignerAuth, RequestsHttpConnection
from opensearchpy.exceptions import OpenSearchException

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

INDEX = "photos"
PAGE_SIZE = 100
URL_TTL_SECONDS = 900
INTENT = "SearchIntent"
KEYWORD_SLOTS = ("KeywordOne", "KeywordTwo")
FREEFORM_SLOT = "QueryText"

SEARCH_PREFIX = re.compile(
    r"^(?:please\s+)?(?:"
    r"(?:show me|find|search for|look for|display|give me)"
    r"(?:\s+(?:(?:me|some|all|the)\s+)*(?:photos?|pictures?|images?)"
    r"(?:\s+(?:of|with|containing|showing))?)?"
    r"|(?:photos?|pictures?|images?)\s+(?:of|with|containing|showing)"
    r")(?:\s+|$)",
    re.IGNORECASE,
)
SEARCH_SUFFIX = re.compile(r"\s+(?:in them|please)$", re.IGNORECASE)
AND_OUTSIDE_QUOTES = re.compile(r'\s+and\s+(?=(?:[^"]*"[^"]*")*[^"]*$)', re.IGNORECASE)
IRREGULAR_PLURALS = {
    "person": "people",
    "man": "men",
    "woman": "women",
    "child": "children",
    "mouse": "mice",
    "goose": "geese",
    "tooth": "teeth",
    "foot": "feet",
    "leaf": "leaves",
    "wolf": "wolves",
}
IRREGULAR_SINGULARS = {plural: single for single, plural in IRREGULAR_PLURALS.items()}

SESSION = boto3.Session()
LEX = SESSION.client(
    "lexv2-runtime",
    config=Config(connect_timeout=3, read_timeout=5, retries={"total_max_attempts": 2}),
)
S3 = SESSION.client("s3", config=Config(signature_version="s3v4"))
SEARCH = OpenSearch(
    hosts=[os.environ["OPENSEARCH_ENDPOINT"].rstrip("/")],
    http_auth=RequestsAWSV4SignerAuth(
        SESSION.get_credentials(), os.environ["AWS_REGION"], "es"
    ),
    connection_class=RequestsHttpConnection,
    use_ssl=True,
    verify_certs=True,
    timeout=5,
    max_retries=1,
)


def search_phrase(text):
    """Take the label phrase out of the search wording supported by the bot."""
    text = text.strip().replace("“", '"').replace("”", '"')
    if not text.startswith('"'):
        text = SEARCH_PREFIX.sub("", text.rstrip("?!").strip(), count=1)
        text = SEARCH_SUFFIX.sub("", text)
    return text.strip()


def slot_text(intent, name):
    # Lex can return null for a slot the user did not fill.
    slot = (intent.get("slots") or {}).get(name) or {}
    value = slot.get("value") or {}
    return value.get("originalValue") or value.get("interpretedValue") or ""


def comparison_text(text):
    return " ".join(text.lower().replace('"', "").split())


def extract_phrase(query):
    session = {
        "botId": os.environ["LEX_BOT_ID"],
        "botAliasId": os.environ["LEX_BOT_ALIAS_ID"],
        "localeId": os.environ.get("LEX_LOCALE_ID", "en_US"),
        "sessionId": uuid4().hex,
    }
    state = LEX.recognize_text(**session, text=query).get("sessionState") or {}
    intent = state.get("intent") or {}
    if intent.get("name") == INTENT:
        captured = [slot_text(intent, name) for name in KEYWORD_SLOTS]
        joined = " and ".join(text for text in captured if text)
        expected = search_phrase(query)
        if joined and comparison_text(joined) == comparison_text(expected):
            # Keep the user's quotation marks if Lex removed them from the slots.
            LOGGER.info("Lex extraction: keyword slots")
            return expected if '"' in expected else joined

    # Let Lex capture an unfamiliar label or the part missed by the keyword slots.
    action = state.get("dialogAction") or {}
    if intent.get("name") != INTENT or action.get("slotToElicit") != FREEFORM_SLOT:
        response = LEX.put_session(
            **session,
            responseContentType="text/plain; charset=utf-8",
            sessionState={
                "dialogAction": {"type": "ElicitSlot", "slotToElicit": FREEFORM_SLOT},
                "intent": {"name": INTENT, "state": "InProgress"},
            },
        )
        if response.get("audioStream") is not None:
            response["audioStream"].close()

    state = LEX.recognize_text(**session, text=query).get("sessionState") or {}
    intent = state.get("intent") or {}
    captured = slot_text(intent, FREEFORM_SLOT) if intent.get("name") == INTENT else ""
    LOGGER.info("Lex extraction: free-form slot populated=%s", bool(captured))
    return search_phrase(captured)


def label_options(phrase):
    """Include the exact normalized label and common English number variants."""
    phrase = phrase.strip().strip('"').strip().lower()
    if not phrase:
        return []
    options = {phrase}
    stem, separator, word = phrase.rpartition(" ")
    if word in IRREGULAR_PLURALS:
        options.add(stem + separator + IRREGULAR_PLURALS[word])
    elif word in IRREGULAR_SINGULARS:
        options.add(stem + separator + IRREGULAR_SINGULARS[word])
    elif re.fullmatch(r"[a-z]{3,}", word):
        variants = []
        if re.search(r"[^aeiou]ies$", word):
            variants.append(word[:-3] + "y")
        if re.search(r"(?:ss|us|x|z|ch|sh)es$", word):
            variants.append(word[:-2])
        elif word.endswith("s") and not word.endswith(("ss", "us", "is")):
            variants.append(word[:-1])
        if re.search(r"[^aeiou]y$", word):
            variants.append(word[:-1] + "ies")
        elif word.endswith(("s", "x", "z", "ch", "sh")):
            variants.append(word + "es")
        else:
            variants.append(word + "s")
        options.update(stem + separator + variant for variant in variants)
    return sorted(options)


def label_query(phrase):
    exact = phrase.strip().strip('"').strip().lower()
    if not exact:
        return None

    parts = AND_OUTSIDE_QUOTES.split(phrase)
    if len(parts) == 1:
        return {"terms": {"labels": label_options(phrase)}}

    groups = [label_options(part) for part in parts]
    if not all(groups):
        return {"term": {"labels": exact}}
    return {
        "bool": {
            "minimum_should_match": 1,
            "should": [
                {"term": {"labels": exact}},
                {"bool": {"filter": [{"terms": {"labels": group}} for group in groups]}},
            ],
        }
    }


def matching_photos(query):
    body = {
        "query": query,
        "size": PAGE_SIZE,
        "sort": [{"bucket": "asc"}, {"objectKey": "asc"}],
        "_source": ["bucket", "objectKey", "labels"],
    }
    while True:
        page = SEARCH.search(
            index=INDEX, body=body, params={"allow_partial_search_results": "false"}
        )
        hits = page["hits"]["hits"]
        for hit in hits:
            yield hit["_source"]
        if len(hits) < PAGE_SIZE:
            return
        body["search_after"] = hits[-1]["sort"]


def json_response(status, payload):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
        "body": json.dumps(payload),
    }


def lambda_handler(event, context):
    params = event.get("queryStringParameters") or {}
    text = params.get("q") or ""
    query = text.strip()
    if not query:
        return json_response(200, {"results": []})
    if len(query) > 1024:
        return json_response(400, {"code": 400, "message": "q must be 1024 characters or fewer."})

    try:
        filter_query = label_query(extract_phrase(query))
        results = []
        if filter_query:
            for photo in matching_photos(filter_query):
                url = S3.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": photo["bucket"], "Key": photo["objectKey"]},
                    ExpiresIn=URL_TTL_SECONDS,
                )
                results.append({"url": url, "labels": photo["labels"]})
        LOGGER.info("Search returned %d photos", len(results))
        return json_response(200, {"results": results})
    except (BotoCoreError, ClientError, OpenSearchException):
        LOGGER.exception("Photo search failed")
        return json_response(500, {"code": 500, "message": "Photo search failed."})
