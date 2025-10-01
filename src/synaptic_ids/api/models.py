from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base


class Prediction(Base):
    """
    ORM model for the 'predictions' table.
    This table stores the output of each prediction made by the model.
    """

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Fields from the prediction result
    label = Column(String, index=True)
    prediction = Column(Integer)
    confidence = Column(Float, nullable=True)

    # Establishes a one-to-one relationship with the TrafficRecord table
    traffic_record = relationship(
        "TrafficRecord",
        back_populates="prediction",
        uselist=False,  # Specifies a one-to-one relationship
        cascade="all, delete-orphan",  # Ensures the related record is deleted
    )


class TrafficRecord(Base):
    """
    ORM model for the 'traffic_records' table.
    This table stores the input features that led to a specific prediction.
    """

    __tablename__ = "traffic_records"

    id = Column(Integer, primary_key=True, index=True)

    # Foreign key linking this record to its corresponding prediction
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=False)

    # Back-reference to the Prediction object
    prediction = relationship("Prediction", back_populates="traffic_record")

    # All fields from the TrafficRecord schema, mapped to database columns.
    srcip = Column(String, nullable=True)
    sport = Column(Integer, nullable=True)
    dstip = Column(String, nullable=True)
    dsport = Column(Integer, nullable=True)
    proto = Column(String, nullable=False)
    state = Column(String, nullable=True)
    dur = Column(Float, nullable=False)
    sbytes = Column(Integer, nullable=False)
    dbytes = Column(Integer, nullable=False)
    sttl = Column(Integer, nullable=False)
    dttl = Column(Integer, nullable=False)
    sloss = Column(Integer, nullable=False)
    dloss = Column(Integer, nullable=False)
    service = Column(String, nullable=False)
    sload = Column(Float, nullable=False)
    dload = Column(Float, nullable=False)
    spkts = Column(Integer, nullable=False)
    dpkts = Column(Integer, nullable=False)
    swin = Column(Integer, nullable=True)
    dwin = Column(Integer, nullable=True)
    stcpb = Column(Integer, nullable=True)
    dtcpb = Column(Integer, nullable=True)
    smean = Column(Integer, nullable=False)
    dmean = Column(Integer, nullable=False)
    trans_depth = Column(Integer, nullable=True)
    response_body_len = Column(Integer, nullable=True)
    sjit = Column(Float, nullable=False)
    djit = Column(Float, nullable=False)
    stime = Column(Integer, nullable=False)
    ltime = Column(Integer, nullable=False)
    sinpkt = Column(Float, nullable=False)
    dinpkt = Column(Float, nullable=False)
    tcprtt = Column(Float, nullable=True)
    synack = Column(Float, nullable=True)
    ackdat = Column(Float, nullable=True)
    is_sm_ips_ports = Column(Integer, nullable=False)
    ct_state_ttl = Column(Integer, nullable=True)
    ct_flw_http_mthd = Column(Integer, nullable=True)
    is_ftp_login = Column(Integer, nullable=True)
    ct_ftp_cmd = Column(Integer, nullable=True)
    ct_srv_src = Column(Integer, nullable=False)
    ct_srv_dst = Column(Integer, nullable=False)
    ct_dst_ltm = Column(Integer, nullable=False)
    ct_src_ltm = Column(Integer, nullable=False)
    ct_src_dport_ltm = Column(Integer, nullable=False)
    ct_dst_sport_ltm = Column(Integer, nullable=False)
    ct_dst_src_ltm = Column(Integer, nullable=False)
    rate = Column(Float)
