from sqlalchemy import text

from app.database import Database


def test_sqlite_initializes_and_session_works(tmp_path) -> None:
    database = Database(f"sqlite:///{tmp_path / 'jarvis.db'}")
    with database.session() as session:
        assert session.execute(text("SELECT 1")).scalar_one() == 1
