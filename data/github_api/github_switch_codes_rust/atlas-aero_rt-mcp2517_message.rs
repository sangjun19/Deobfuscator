// Repository: atlas-aero/rt-mcp2517
// File: src/tests/message.rs

use crate::message::{Can20, CanFd, MessageError, TxMessage, DLC};
use bytes::Bytes;
use embedded_can::Id;
use embedded_can::{ExtendedId, StandardId};

const EXTENDED_ID: u32 = 0x14C92A2B;

const STANDARD_ID: u16 = 0x6A5;

#[test]
fn test_extended_id() {
    let payload_bytes = Bytes::copy_from_slice(&[0u8; 8]);
    let extended_id = ExtendedId::new(EXTENDED_ID).unwrap();

    let msg_type = Can20::<8> {};

    let message = TxMessage::new(msg_type, payload_bytes, Id::Extended(extended_id)).unwrap();

    assert!(message.header.identifier_extension_flag());
    assert_eq!(message.header.extended_identifier(), 0b01_0010_1010_0010_1011);
    assert_eq!(message.header.standard_identifier(), 0b101_0011_0010);
}

#[test]
fn test_standard_id() {
    let payload_bytes = Bytes::copy_from_slice(&[0u8; 8]);
    let standard_id = StandardId::new(STANDARD_ID).unwrap();

    let msg_type = Can20::<8> {};

    let message = TxMessage::new(msg_type, payload_bytes, Id::Standard(standard_id)).unwrap();

    assert!(!message.header.identifier_extension_flag());
    assert_eq!(message.header.extended_identifier(), 0b00_0000_0000_0000_0000);
    assert_eq!(message.header.standard_identifier(), 0b110_1010_0101);
}

#[test]
fn test_dlc_success() {
    let payload_bytes = Bytes::copy_from_slice(&[0u8; 13]);
    let standard_id = StandardId::new(STANDARD_ID).unwrap();

    let msg_type = CanFd::<16> { bitrate_switch: false };

    let message = TxMessage::new(msg_type, payload_bytes, Id::Standard(standard_id)).unwrap();

    assert_eq!(message.header.data_length_code(), DLC::Sixteen);
    assert!(message.header.fd_frame());

    let header_bytes = message.header.into_bytes();

    assert_eq!(header_bytes[7], 0b1000_1010);
}

#[test]
fn test_dlc_error() {
    let data_2_0 = [0u8; 10];
    let data_fd = [0u8; 65];

    let payload_bytes_2_0 = Bytes::copy_from_slice(&data_2_0);
    let payload_bytes_fd = Bytes::copy_from_slice(&data_fd);

    let can_msg_20 = Can20::<8> {};
    let can_msg_fd = CanFd::<64> { bitrate_switch: false };

    let standard_id = StandardId::new(STANDARD_ID).unwrap();

    let message_2_0 = TxMessage::new(can_msg_20, payload_bytes_2_0, Id::Standard(standard_id));
    let message_fd = TxMessage::new(can_msg_fd, payload_bytes_fd, Id::Standard(standard_id));

    assert_eq!(message_2_0.unwrap_err(), MessageError::InvalidLength(10));
    assert_eq!(message_fd.unwrap_err(), MessageError::InvalidLength(65));
}

#[test]
fn test_message_size_divisible_by_four_error() {
    let data_2_0 = [0u8; 6];
    let data_fd = [0u8; 26];

    let payload_bytes_2_0 = Bytes::copy_from_slice(&data_2_0);
    let payload_bytes_fd = Bytes::copy_from_slice(&data_fd);

    let can_msg_20 = Can20::<6> {};
    let can_msg_fd = CanFd::<26> { bitrate_switch: false };

    let standard_id = StandardId::new(STANDARD_ID).unwrap();

    let message_2_0 = TxMessage::new(can_msg_20, payload_bytes_2_0, Id::Standard(standard_id));
    let message_fd = TxMessage::new(can_msg_fd, payload_bytes_fd, Id::Standard(standard_id));

    assert_eq!(message_2_0.unwrap_err(), MessageError::InvalidTypeSize(6));
    assert_eq!(message_fd.unwrap_err(), MessageError::InvalidTypeSize(26));
}

#[test]
fn test_payload_greater_than_generic_type_args() {
    let data_2_0 = [0u8; 5];
    let data_fd = [0u8; 23];

    let payload_bytes_2_0 = Bytes::copy_from_slice(&data_2_0);
    let payload_bytes_fd = Bytes::copy_from_slice(&data_fd);

    let can_msg_20 = Can20::<4> {};
    let can_msg_fd = CanFd::<20> { bitrate_switch: false };

    let standard_id = StandardId::new(STANDARD_ID).unwrap();

    let message_2_0 = TxMessage::new(can_msg_20, payload_bytes_2_0, Id::Standard(standard_id));
    let message_fd = TxMessage::new(can_msg_fd, payload_bytes_fd, Id::Standard(standard_id));

    assert_eq!(message_2_0.unwrap_err(), MessageError::InvalidLength(5));
    assert_eq!(message_fd.unwrap_err(), MessageError::InvalidLength(23));
}

#[test]
fn test_get_payload() {
    let payload_bytes = Bytes::copy_from_slice(&[1u8; 8]);
    let standard_id = StandardId::new(STANDARD_ID).unwrap();

    let msg_type = Can20::<8> {};

    let message = TxMessage::new(msg_type, payload_bytes, Id::Standard(standard_id)).unwrap();

    assert_eq!(message.get_payload(), &[1u8; 8]);
}
