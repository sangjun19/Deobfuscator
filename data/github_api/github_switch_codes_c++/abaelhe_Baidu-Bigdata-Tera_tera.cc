// Copyright (c) 2015, Baidu.com, Inc. All Rights Reserved
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "tera.h"

#include <limits>

namespace tera {

static const char* strerr(ErrorCode::ErrorCodeType type) {
    const char* ret = "Unknown error";
    switch (type) {
    case ErrorCode::kOK:
        ret = "OK";
        break;
    case ErrorCode::kNotFound:
        ret = "Not Found";
        break;
    case ErrorCode::kBadParam:
        ret = "Bad Parameter";
        break;
    case ErrorCode::kSystem:
        ret = "SystemError";
        break;
    case ErrorCode::kTimeout:
        ret = "Timeout";
        break;
    case ErrorCode::kBusy:
        ret = "SystemBusy";
        break;
    case ErrorCode::kNoQuota:
        ret = "UserNoQuota";
        break;
    case ErrorCode::kNoAuth:
        ret = "UserUnauthorized";
        break;
    case ErrorCode::kNotImpl:
        ret = "Not Implement";
        break;
    case ErrorCode::kTxnFail:
        ret = "TransactionFail";
        break;
    default:
        ret = "UnkownError";
    }
    return ret;
}

ErrorCode::ErrorCode() : err_(kOK) {
}

void ErrorCode::SetFailed(ErrorCodeType err, const std::string& reason) {
    err_ = err;
    reason_ = reason;
}

std::string ErrorCode::ToString() const {
    std::string ret;
    ret.append("type [");
    ret.append(strerr(err_));
    ret.append("], reason [");
    ret.append(reason_);
    ret.append("].");
    return ret;
}

std::string ErrorCode::GetReason() const {
    return reason_;
}

ErrorCode::ErrorCodeType ErrorCode::GetType() const {
    return err_;
}

const char* strerr(ErrorCode error_code) {
    return strerr(error_code.GetType());
}

const int64_t kLatestTimestamp = std::numeric_limits<long int>::max();
const int64_t kOldestTimestamp = std::numeric_limits<long int>::min();

}
