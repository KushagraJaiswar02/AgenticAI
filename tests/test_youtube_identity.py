import pytest

from app.youtube_identity import video_identity, video_url


@pytest.mark.parametrize("reference", [
    "AaBbCcDdE-1", "youtube.com/watch?v=AaBbCcDdE-1&t=65&si=opaque",
    "https://youtu.be/AaBbCcDdE-1#t=12", "https://m.youtube.com/shorts/AaBbCcDdE-1",
    "https://www.youtube.com/embed/AaBbCcDdE-1", "https://youtube.com:443/live/AaBbCcDdE-1",
])
def test_video_identity(reference):
    assert video_identity(reference) == ("youtube", "AaBbCcDdE-1")
    assert video_url(reference) == "https://www.youtube.com/watch?v=AaBbCcDdE-1"
    assert video_identity(reference) != video_identity("aaBbCcDdE-1")


@pytest.mark.parametrize("reference", [
    "short", "AaBbCcDdE!1", "https://youtube.com/watch?v=AaBbCcDdE-1&%76=AaBbCcDdE-1",
    "https://youtube.com.evil.test/watch?v=AaBbCcDdE-1", "https://user@youtube.com/watch?v=AaBbCcDdE-1",
    "https://youtube.com:444/watch?v=AaBbCcDdE-1", "https://youtube.com:/watch?v=AaBbCcDdE-1",
    "https://youtube.com/watch?v=AaBbCcDdE-1&t=%ZZ", "https://youtu.be/AaBbCcDdE-1/extra",
    "https://youtu.be/AaBbCcDdE-1%2Fextra", "https://youtube.com/playlist?list=AaBbCcDdE-1",
    "https://[broken/watch?v=AaBbCcDdE-1", "https://youtube.com\\@other.test/watch?v=AaBbCcDdE-1",
])
def test_rejects_invalid_reference(reference):
    with pytest.raises(ValueError, match="Invalid YouTube"):
        video_identity(reference)
