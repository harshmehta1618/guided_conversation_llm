from pydantic import BaseModel, Field
from typing import List

class UserHealthProfile(BaseModel):
    Age: int = Field(default=-1)
    Weight: float = Field(default=-1)
    Average_Sleeping_Hours: float = Field(default=-1)
    Has_Diabetes: int = Field(default=-1)
    Has_High_Blood_Pressure: int = Field(default=-1)
    Daily_Exercise: int = Field(default=-1)
    # New fields from conversations
    Medications: List[str] = Field(default_factory=list)
    Has_Asthma: int = Field(default=-1)
    Has_Pneumonia_History: int = Field(default=-1)
    Has_Shoulder_Injury: int = Field(default=-1)
    Has_Tremors: int = Field(default=-1)
    Has_Mouth_Sores: int = Field(default=-1)
    Has_Nausea_Vomiting: int = Field(default=-1)
    Has_Stomach_Pain: int = Field(default=-1)
    Cough_With_Phlegm: int = Field(default=-1)
    White_Coat_Hypertension: int = Field(default=-1)
    Degenerative_Changes: int = Field(default=-1)
    Wears_Dentures: int = Field(default=-1)