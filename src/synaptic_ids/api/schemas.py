from typing import List, Optional, Dict
from pydantic import BaseModel, Field, ConfigDict

# ====================================================================
#                 SCHEMAS FOR ML MODEL PREDICTION
# ====================================================================


class TrafficRecord(BaseModel):
    """
    Represents a single network traffic record based on the UNSW-NB15 dataset.
    Each attribute corresponds to a feature of the network traffic.
    These fields are used as input for the IDS prediction model.
    """

    srcip: Optional[str] = Field(None, description="Source IP address")
    sport: Optional[int] = Field(None, description="Source port")
    dstip: Optional[str] = Field(None, description="Destination IP address")
    dsport: Optional[int] = Field(None, description="Destination port")
    proto: str = Field(..., description="Transaction protocol")
    state: Optional[str] = Field(
        None, description="Transaction state and its dependent"
    )
    dur: float = Field(..., description="Total transaction duration")
    sbytes: int = Field(..., description="Bytes from sender to receiver")
    dbytes: int = Field(..., description="Bytes from receiver to sender")
    sttl: int = Field(..., description="TTL from sender to receiver")
    dttl: int = Field(..., description="TTL from receiver to sender")
    sloss: int = Field(..., description="Packets lost from sender to receiver")
    dloss: int = Field(..., description="Packets lost from receiver to sender")
    service: str = Field(..., description="Destination service http, ftp, etc.")
    sload: float = Field(
        ..., description="Bit rate from sender to receiver in bits/sec"
    )
    dload: float = Field(
        ..., description="Bit rate from receiver to sender in bits/sec"
    )
    spkts: int = Field(..., description="Packet count from sender to receiver")
    dpkts: int = Field(..., description="Packet count from receiver to sender")
    swin: Optional[int] = Field(None, description="TCP send window size")
    dwin: Optional[int] = Field(None, description="TCP destination window size")
    stcpb: Optional[int] = Field(None, description="Source TCP base sequence number")
    dtcpb: Optional[int] = Field(
        None, description="Destination TCP base sequence number"
    )
    smean: int = Field(..., description="Average packet size from sender")
    dmean: int = Field(..., description="Average packet size from receiver")
    trans_depth: Optional[int] = Field(None, description="Depth of an HTTP transaction")
    response_body_len: Optional[int] = Field(
        None, description="HTTP response body size"
    )
    sjit: float = Field(..., description="Source jitter (ms)")
    djit: float = Field(..., description="Destination jitter (ms)")
    stime: int = Field(..., description="Start timestamp")
    ltime: int = Field(..., description="End timestamp")
    sinpkt: float = Field(..., description="Average time between source packets")
    dinpkt: float = Field(..., description="Average time between destination packets")
    tcprtt: Optional[float] = Field(None, description="TCP connection RTT")
    synack: Optional[float] = Field(None, description="Time from SYN to ACK")
    ackdat: Optional[float] = Field(None, description="Time from ACK to data")
    is_sm_ips_ports: int = Field(
        ..., description="If source and destination IP/port are the same"
    )
    ct_state_ttl: Optional[int] = Field(
        None, description="Connection count with the same state and TTL"
    )
    ct_flw_http_mthd: Optional[int] = Field(
        None, description="Flow count with HTTP methods"
    )
    is_ftp_login: Optional[int] = Field(None, description="If there is an FTP login")
    ct_ftp_cmd: Optional[int] = Field(None, description="Count of FTP commands")
    ct_srv_src: int = Field(
        ..., description="Connection count to the same source service"
    )
    ct_srv_dst: int = Field(
        ..., description="Connection count to the same destination service"
    )
    ct_dst_ltm: int = Field(
        ..., description="Connection count to the same destination IP"
    )
    ct_src_ltm: int = Field(..., description="Connection count to the same source IP")
    ct_src_dport_ltm: int = Field(
        ..., description="Connection count to the same source destination port"
    )
    ct_dst_sport_ltm: int = Field(
        ..., description="Connection count to the same destination source port"
    )
    ct_dst_src_ltm: int = Field(
        ..., description="Connection count to the same source/destination IP pair"
    )
    rate: float = Field(..., description="Packets per second")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "proto": "tcp",
                "state": "FIN",
                "dur": 0.088159,
                "sbytes": 568,
                "dbytes": 3130,
                "sttl": 62,
                "dttl": 252,
                "sloss": 7,
                "dloss": 5,
                "service": "-",
                "sload": 45554.402344,
                "dload": 243763.421875,
                "spkts": 18,
                "dpkts": 14,
                "smean": 32,
                "dmean": 224,
                "sjit": 23.477833,
                "djit": 1.400333,
                "stime": 1421927414,
                "ltime": 1421927414,
                "sinpkt": 5.185823,
                "dinpkt": 6.209231,
                "is_sm_ips_ports": 0,
                "ct_srv_src": 3,
                "ct_srv_dst": 2,
                "ct_dst_ltm": 1,
                "ct_src_ltm": 1,
                "ct_src_dport_ltm": 1,
                "ct_dst_sport_ltm": 1,
                "ct_dst_src_ltm": 1,
                "rate": 351.638062,
                "response_body_len": 0,
                "tcprtt": 0.057393,
                "synack": 0.013233,
                "ackdat": 0.04416,
                "ct_state_ttl": 0,
                "ct_flw_http_mthd": 0,
                "is_ftp_login": 0,
                "ct_ftp_cmd": 0,
                "trans_depth": 0,
                "swin": 255,
                "dwin": 255,
                "stcpb": 133588210,
                "dtcpb": 357039015,
            }
        }
    )


class PredictionInput(BaseModel):
    """
    Schema for the prediction input, containing a list of traffic records.
    """

    session_id: str = Field(
        ..., description="A unique identifier for the user session."
    )
    records: List[TrafficRecord] = Field(
        ..., description="List of traffic records to classify"
    )


class PredictionResult(BaseModel):
    """
    Represents the prediction result for a single traffic record.
    """

    label: str = Field(
        ..., description="The predicted class label (e.g., 'Normal' or attack type)"
    )
    prediction: int = Field(..., description="The numeric value of the predicted class")
    confidence: Optional[float] = Field(
        None, description="The confidence score of the prediction (probability)"
    )
    probabilities: Optional[Dict[str, float]] = Field(
        None, description="Probabilities for each class (for multiclass mode)"
    )


class PredictionResponse(BaseModel):
    """
    Schema for the API response, containing a list of prediction results.
    """

    predictions: List[PredictionResult] = Field(
        ..., description="List of prediction results"
    )


# ====================================================================
#              SCHEMAS FOR DATABASE CRUD OPERATIONS
# ====================================================================


class PredictionRecord(PredictionResult):
    """
    Represents a full prediction record as read from the database,
    including its unique ID.
    Inherits 'label', 'prediction', 'confidence', and 'probabilities'
    from PredictionResult.
    """

    id: int
    model_config = ConfigDict(from_attributes=True)
