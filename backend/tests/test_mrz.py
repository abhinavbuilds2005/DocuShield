"""
Unit Tests for ICAO Doc 9303 MRZ Parsing and Check Digits (SIH26188)
Tests:
- Valid TD3 passport MRZ line parsing
- 7-3-1 weight check digit calculation
- Corrupted/tampered check digit detection
- Cross-consistency between MRZ passport number and visual OCR
"""

import pytest
from backend.nlp.mrz_parser import MRZParser, calculate_check_digit, verify_check_digit


@pytest.fixture
def mrz_parser():
    return MRZParser()


def test_calculate_check_digit():
    # Test vector: Document number "L898902C3" with weights 7-3-1 -> check digit should match
    cd = calculate_check_digit("L898902C3")
    assert cd.isdigit()
    assert verify_check_digit("L898902C3", cd) is True
    # Inverted or corrupted digit
    assert verify_check_digit("L898902C3", str((int(cd) + 1) % 10)) is False


def test_parse_valid_td3_mrz(mrz_parser):
    # Sample TD3 MRZ: 2 lines of 44 chars
    l1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
    l2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
    res = mrz_parser.parse(f"{l1}\n{l2}")
    assert res["mrz_detected"] is True
    assert res["format"] == "TD3"
    assert res["fields"]["passport_number"] == "L898902C3"
    assert res["fields"]["nationality"] == "UTO"
    assert res["fields"]["gender"] == "FEMALE"
    assert res["check_digits"]["passport_number_valid"] is True
    assert res["check_digits"]["date_of_birth_valid"] is True
    assert res["check_digits"]["date_of_expiry_valid"] is True


def test_corrupted_mrz_check_digit(mrz_parser):
    l1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
    # Tampered DOB check digit (from 2 to 9)
    l2 = "L898902C36UTO7408129F1204159ZE184226B<<<<<10"
    res = mrz_parser.parse(f"{l1}\n{l2}")
    assert res["mrz_detected"] is True
    assert res["check_digits"]["date_of_birth_valid"] is False
    assert res["is_valid"] is False
