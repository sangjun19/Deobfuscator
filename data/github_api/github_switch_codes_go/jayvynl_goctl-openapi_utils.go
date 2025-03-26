// Repository: jayvynl/goctl-openapi
// File: oas3/utils.go

package oas3

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/getkin/kin-openapi/openapi3"
	"github.com/jayvynl/goctl-openapi/constant"
)

var (
	pathParamRe    = regexp.MustCompile(`/:([^/]+)`)
	ErrInvalidType = fmt.Errorf("invalid type")
)

func ConvertPath(path string) string {
	return pathParamRe.ReplaceAllString(path, `/{$1}`)
}

func GetProperty(properties map[string]string, key string) string {
	v, err := strconv.Unquote(properties[key])
	if err == nil {
		return v
	}
	return properties[key]
}

// https://pkg.go.dev/github.com/go-playground/validator/v10#hdr-Using_Validator_Tags
func UnescapeValidateString(s string) string {
	return strings.ReplaceAll(strings.ReplaceAll(s, "0x2C", ","), "0x7C", "|")
}

// GetMapValueType map[[2]string][]string -> []string
func GetMapValueType(typ string) (string, error) {
	if !strings.HasPrefix(typ, "map[") {
		return "", ErrInvalidType
	}

	var level int
	for i := 4; i < len(typ)-1; i++ {
		if typ[i] == '[' {
			level++
		} else if typ[i] == ']' {
			if level == 0 {
				return typ[i+1:], nil
			}
			level--
		}
	}
	return "", ErrInvalidType
}

// MergeRequired merge schema required fields
func MergeRequired(rs ...[]string) []string {
	if len(rs) == 0 {
		return nil
	}
	if len(rs) == 1 {
		return rs[0]
	}

	merged := make([]string, len(rs[0]))
	copy(merged, rs[0])
	for i := 1; i < len(rs); i++ {
	out:
		for _, f := range rs[i] {
			for _, mf := range merged {
				if f == mf {
					continue out
				}
			}
			merged = append(merged, f)
		}
	}
	return merged
}

func ParseValue(typ string, format string, s string) (interface{}, error) {
	switch typ {
	case openapi3.TypeBoolean:
		return strconv.ParseBool(s)
	case openapi3.TypeInteger:
		return ParseInteger(format, s)
	case openapi3.TypeNumber:
		return ParseNumber(format, s)
	case openapi3.TypeString:
		return s, nil
	default:
		return nil, fmt.Errorf("can't parse type \"%s\"", typ)
	}
}

func ParseInteger(format string, s string) (float64, error) {
	var bits int

	switch format {
	case constant.FormatInt8, constant.FormatUint8:
		bits = 8
	case constant.FormatInt16, constant.FormatUint16:
		bits = 16
	case constant.FormatInt32, constant.FormatUint32:
		bits = 32
	default:
		bits = 64
	}
	if IsUint(format) {
		v, err := strconv.ParseUint(s, 10, bits)
		return float64(v), err
	}
	v, err := strconv.ParseInt(s, 10, bits)
	return float64(v), err
}

func ParseNumber(format string, s string) (float64, error) {
	if format == constant.FormatFloat {
		return strconv.ParseFloat(s, 32)
	}
	return strconv.ParseFloat(s, 64)
}

func IsUint(format string) bool {
	return format == constant.FormatUint || format == constant.FormatUint8 || format == constant.FormatUint16 ||
		format == constant.FormatUint32 || format == constant.FormatUint64
}
