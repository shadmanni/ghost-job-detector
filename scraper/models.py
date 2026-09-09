from datetime import datetime
import os
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()


class JobPosting(Base):
    __tablename__ = "job_postings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    company = Column(String(255), nullable=False, index=True)
    title = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    url = Column(String(1024), nullable=True)
    source = Column(String(100), nullable=True, index=True)
    posted_date = Column(DateTime, nullable=True)
    scraped_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    salary_listed = Column(String(255), nullable=True)
    repost_of_id = Column(Integer, ForeignKey("job_postings.id"), nullable=True)

    # Relationships
    repost_of = relationship("JobPosting", remote_side=[id], backref="reposts")
    scores = relationship("GhostScore", back_populates="job_posting", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<JobPosting(id={self.id}, company='{self.company}', title='{self.title}')>"


class CompanyReview(Base):
    __tablename__ = "company_reviews"

    id = Column(Integer, primary_key=True, autoincrement=True)
    company = Column(String(255), nullable=False, index=True)
    source = Column(String(100), nullable=True, index=True)
    text = Column(Text, nullable=False)
    posted_at = Column(DateTime, nullable=True)
    scraped_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<CompanyReview(id={self.id}, company='{self.company}', source='{self.source}')>"


class GhostScore(Base):
    __tablename__ = "ghost_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_posting_id = Column(Integer, ForeignKey("job_postings.id"), nullable=False, index=True)
    genericness_score = Column(Float, nullable=True)
    vagueness_score = Column(Float, nullable=True)
    repost_score = Column(Float, nullable=True)
    urgency_score = Column(Float, nullable=True)
    bert_score = Column(Float, nullable=True)
    sentiment_score = Column(Float, nullable=True)
    final_score = Column(Float, nullable=True)
    computed_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationship
    job_posting = relationship("JobPosting", back_populates="scores")

    def __repr__(self):
        return f"<GhostScore(id={self.id}, job_posting_id={self.job_posting_id}, final_score={self.final_score})>"


def get_db_engine(db_path: str = "data/ghostjobs.db"):
    # Ensure parent directory exists
    dir_path = os.path.dirname(db_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    engine_url = f"sqlite:///{db_path}"
    return create_engine(engine_url, echo=False)


def init_db(db_path: str = "data/ghostjobs.db"):
    engine = get_db_engine(db_path)
    Base.metadata.create_all(bind=engine)
    return engine


def get_session(db_path: str = "data/ghostjobs.db"):
    engine = get_db_engine(db_path)
    Session = sessionmaker(bind=engine)
    return Session()
