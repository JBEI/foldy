"""Database unit tests."""
import pytest
from sqlalchemy.orm.exc import ObjectDeletedError

from app.database import Column, PkModel, db
from app.models import User


@pytest.mark.usefixtures("db")
class TestCRUDMixin:
    """CRUDMixin tests."""

    def test_create(self):
        """Test CRUD create."""
        user = User.create(email="foo@bar.com")
        assert User.get_by_id(user.id).email == "foo@bar.com"

    def test_create_save(self):
        """Test CRUD create with save."""
        user = User("foo@bar.com")
        user.save()
        assert User.get_by_id(user.id) is not None

    def test_delete_with_commit(self):
        """Test CRUD delete with commit."""
        user = User("foo@bar.com")
        user.save()
        user.delete(commit=True)
        assert User.get_by_id(user.id) is None

    def test_delete_without_commit_cannot_access(self):
        """Test CRUD delete without commit."""
        user = User("foo@bar.com")
        user.save()
        user.delete(commit=False)
        with pytest.raises(ObjectDeletedError):
            User.get_by_id(user.id)

    @pytest.mark.parametrize(
        "commit,expected", [(True, "baz@bar.com"), (False, "foo@bar.com")]
    )
    def test_update(self, commit, expected, db):
        """Test CRUD update with and without commit."""
        user = User(email="foo@bar.com")
        user.save()
        user.update(commit=commit, email="baz@bar.com")
        retrieved = db.session.execute("""select * from users""").fetchone()
        assert retrieved.email == expected


class TestPkModel:
    """PkModel tests."""

    def test_get_by_id_wrong_type(self):
        """Test get_by_id returns None for non-numeric argument."""
        assert User.get_by_id("xyz") is None
