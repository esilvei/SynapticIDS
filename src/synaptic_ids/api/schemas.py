from typing import List, Optional
from pydantic import BaseModel, Field


class UNSWNB15FeatureSchema(BaseModel):
    """
    Schema for the raw features of the UNSW-NB15 dataset.
    """

    dur: float
    proto: str
    service: str
    state: str
    spkts: int
    dpkts: int
    sbytes: int
    dbytes: int
    rate: float
    sttl: int
    dttl: int
    sload: float
    dload: float
    sloss: int
    dloss: int
    sinpkt: float
    dinpkt: float
    sjit: float
    djit: float
    swin: int
    stcpb: int
    dtcpb: int
    dwin: int
    tcprtt: float
    synack: float
    ackdat: float
    smean: int
    dmean: int
    trans_depth: int
    response_body_len: int
    ct_srv_src: int
    ct_state_ttl: int
    ct_dst_ltm: int
    ct_src_dport_ltm: int
    ct_dst_sport_ltm: int
    ct_dst_src_ltm: int
    is_ftp_login: int
    ct_ftp_cmd: int
    ct_flw_http_mthd: int
    ct_src_ltm: int
    ct_srv_dst: int
    is_sm_ips_ports: int
    attack_cat: Optional[str] = None
    label: int


class ModelInputSchema(BaseModel):
    """
    Schema for the model input data after preprocessing.
    """

    features: List[float] = Field(
        ..., description="Vector of features after preprocessing and feature selection."
    )


class PredictionOutputSchema(BaseModel):
    """
    Schema for the model's prediction output.
    """

    prediction: int = Field(..., description="The predicted class by the model.")
    probability: float = Field(
        ..., description="The probability of the predicted class."
    )
