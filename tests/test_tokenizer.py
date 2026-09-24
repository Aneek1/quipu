from quipu.tokenizer import Tokenizer


def test_vocab_size_matches_the_config():
    assert Tokenizer().vocab_size == 50257


def test_round_trips_text():
    tok = Tokenizer()
    text = "The quipu encoded records in knotted cords."
    assert tok.decode(tok.encode(text)) == text


def test_eot_is_the_documented_id():
    # 50256 is <|endoftext|> in the GPT-2 vocabulary. data.py separates documents
    # with it, so a change here silently changes the shard format.
    assert Tokenizer().eot == 50256


def test_every_token_id_fits_in_uint16():
    # Shards are uint16. A vocabulary above 65535 would wrap silently.
    assert Tokenizer().vocab_size <= 65536


def test_endoftext_literal_in_text_encodes_without_raising_and_is_not_the_eot_token():
    # encode_ordinary treats "<|endoftext|>" appearing in a document as ordinary
    # text rather than the special token. If encode() were switched to encode()
    # with special-token parsing enabled, this literal would either raise or be
    # collapsed into token 50256 -- silently corrupting the shard's document
    # boundaries.
    tok = Tokenizer()
    text = "before <|endoftext|> after"
    ids = tok.encode(text)
    assert tok.eot not in ids
