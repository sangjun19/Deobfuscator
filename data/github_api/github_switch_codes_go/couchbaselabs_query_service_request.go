// Repository: couchbaselabs/query
// File: server/http/service_request.go

//  Copyright (c) 2014 Couchbase, Inc.
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
//  except in compliance with the License. You may obtain a copy of the License at
//    http://www.apache.org/licenses/LICENSE-2.0
//  Unless required by applicable law or agreed to in writing, software distributed under the
//  License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
//  either express or implied. See the License for the specific language governing permissions
//  and limitations under the License.

package http

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/couchbaselabs/query/datastore"
	"github.com/couchbaselabs/query/errors"
	"github.com/couchbaselabs/query/plan"
	"github.com/couchbaselabs/query/server"
	"github.com/couchbaselabs/query/timestamp"
	"github.com/couchbaselabs/query/value"
)

const MAX_REQUEST_BYTES = 1 << 20

type httpRequest struct {
	server.BaseRequest
	resp         http.ResponseWriter
	req          *http.Request
	writer       responseDataManager
	httpRespCode int
	resultCount  int
	resultSize   int
	errorCount   int
	warningCount int
}

func newHttpRequest(resp http.ResponseWriter, req *http.Request, bp BufferPool) *httpRequest {
	var httpArgs httpRequestArgs
	var err errors.Error

	e := req.ParseForm()
	if e != nil {
		err = errors.NewServiceErrorBadValue(e, "request form")
	}

	if err != nil && req.Method != "GET" && req.Method != "POST" {
		err = errors.NewServiceErrorHTTPMethod(req.Method)
	}

	if err == nil {
		httpArgs, err = getRequestParams(req)
	}

	var statement string
	if err == nil {
		statement, err = httpArgs.getStatement()
	}

	var prepared *plan.Prepared
	if err == nil {
		prepared, err = getPrepared(httpArgs)
	}

	if err == nil && statement == "" && prepared == nil {
		err = errors.NewServiceErrorMissingValue("statement or prepared")
	}

	var namedArgs map[string]value.Value
	if err == nil {
		namedArgs, err = httpArgs.getNamedArgs()
	}

	var positionalArgs value.Values
	if err == nil {
		positionalArgs, err = httpArgs.getPositionalArgs()
	}

	var namespace string
	if err == nil {
		namespace, err = httpArgs.getString(NAMESPACE, "")
	}

	var timeout time.Duration
	if err == nil {
		timeout, err = httpArgs.getDuration(TIMEOUT)
	}

	var readonly value.Tristate
	if err == nil {
		readonly, err = httpArgs.getTristate(READONLY)
	}
	if err == nil && readonly == value.FALSE && req.Method == "GET" {
		err = errors.NewServiceErrorReadonly(
			fmt.Sprintf("%s=false cannot be used with HTTP GET method.", READONLY))
	}

	var metrics value.Tristate
	if err == nil {
		metrics, err = httpArgs.getTristate(METRICS)
	}

	var format Format
	if err == nil {
		format, err = getFormat(httpArgs)
	}

	if err == nil && format != JSON {
		err = errors.NewServiceErrorNotImplemented("format", format.String())
	}

	var signature value.Tristate
	if err == nil {
		signature, err = httpArgs.getTristate(SIGNATURE)
	}

	var compression Compression
	if err == nil {
		compression, err = getCompression(httpArgs)
	}

	if err == nil && compression != NONE {
		err = errors.NewServiceErrorNotImplemented("compression", compression.String())
	}

	var encoding Encoding
	if err == nil {
		encoding, err = getEncoding(httpArgs)
	}

	if err == nil && encoding != UTF8 {
		err = errors.NewServiceErrorNotImplemented("encoding", encoding.String())
	}

	var pretty value.Tristate
	if err == nil {
		pretty, err = httpArgs.getTristate(PRETTY)
	}

	if err == nil && pretty == value.FALSE {
		err = errors.NewServiceErrorNotImplemented("pretty", "false")
	}

	var consistency *scanConfigImpl

	if err == nil {
		consistency, err = getScanConfiguration(httpArgs)
	}

	var creds datastore.Credentials
	if err == nil {
		creds, err = getCredentials(httpArgs, req.URL.User, req.Header["Authorization"])
	}

	client_id := ""
	if err == nil {
		client_id, err = httpArgs.getString(CLIENT_CONTEXT_ID, "")
	}

	base := server.NewBaseRequest(statement, prepared, namedArgs, positionalArgs,
		namespace, readonly, metrics, signature, consistency, client_id, creds)

	rv := &httpRequest{
		BaseRequest: *base,
		resp:        resp,
		req:         req,
	}

	rv.SetTimeout(rv, timeout)

	rv.writer = NewBufferedWriter(rv, bp)

	// Limit body size in case of denial-of-service attack
	req.Body = http.MaxBytesReader(resp, req.Body, MAX_REQUEST_BYTES)

	// Abort if client closes connection
	closeNotify := resp.(http.CloseNotifier).CloseNotify()
	go func() {
		<-closeNotify
		rv.Stop(server.TIMEOUT)
	}()

	if err != nil {
		rv.Fail(err)
	}

	return rv
}

const ( // Request argument names
	READONLY          = "readonly"
	METRICS           = "metrics"
	NAMESPACE         = "namespace"
	TIMEOUT           = "timeout"
	ARGS              = "args"
	PREPARED          = "prepared"
	STATEMENT         = "statement"
	FORMAT            = "format"
	ENCODING          = "encoding"
	COMPRESSION       = "compression"
	SIGNATURE         = "signature"
	PRETTY            = "pretty"
	SCAN_CONSISTENCY  = "scan_consistency"
	SCAN_WAIT         = "scan_wait"
	SCAN_VECTOR       = "scan_vector"
	CREDS             = "creds"
	CLIENT_CONTEXT_ID = "client_context_id"
)

func getPrepared(a httpRequestArgs) (*plan.Prepared, errors.Error) {
	prepared_field, err := a.getValue(PREPARED)
	if err != nil || prepared_field == nil {
		return nil, err
	}

	prepared, e := plan.PreparedCache().GetPrepared(prepared_field)
	if e != nil {
		return nil, errors.NewServiceErrorBadValue(e, PREPARED)
	}
	if prepared != nil {
		return prepared, nil
	}

	prepared = &plan.Prepared{}
	json_bytes, e := prepared_field.MarshalJSON()
	if e != nil {
		return nil, errors.NewServiceErrorBadValue(e, PREPARED)
	}

	e = prepared.UnmarshalJSON(json_bytes)
	if e != nil {
		return nil, errors.NewServiceErrorBadValue(e, PREPARED)
	}

	e = plan.PreparedCache().AddPrepared(prepared)
	if e != nil {
		return nil, errors.NewServiceErrorBadValue(e, PREPARED)
	}

	return prepared, nil
}

func getCompression(a httpRequestArgs) (Compression, errors.Error) {
	var compression Compression

	compression_field, err := a.getString(COMPRESSION, "NONE")
	if err == nil && compression_field != "" {
		compression = newCompression(compression_field)
		if compression == UNDEFINED_COMPRESSION {
			err = errors.NewServiceErrorUnrecognizedValue(COMPRESSION, compression_field)
		}
	}
	return compression, err
}

func getScanConfiguration(a httpRequestArgs) (*scanConfigImpl, errors.Error) {
	var sc scanConfigImpl

	scan_consistency_field, err := a.getString(SCAN_CONSISTENCY, "NOT_BOUNDED")
	if err == nil {
		sc.scan_level = newScanConsistency(scan_consistency_field)
		if sc.scan_level == server.UNDEFINED_CONSISTENCY {
			err = errors.NewServiceErrorUnrecognizedValue(SCAN_CONSISTENCY, scan_consistency_field)
		}
	}
	if err == nil {
		sc.scan_wait, err = a.getDuration(SCAN_WAIT)
	}
	if err == nil {
		sc.scan_vector, err = a.getScanVector()
	}
	if err == nil && sc.scan_level == server.AT_PLUS && sc.scan_vector == nil {
		err = errors.NewServiceErrorMissingValue(SCAN_VECTOR)
	}
	return &sc, err
}

func getEncoding(a httpRequestArgs) (Encoding, errors.Error) {
	var encoding Encoding

	encoding_field, err := a.getString(ENCODING, "UTF-8")
	if err == nil && encoding_field != "" {
		encoding = newEncoding(encoding_field)
		if encoding == UNDEFINED_ENCODING {
			err = errors.NewServiceErrorUnrecognizedValue(ENCODING, encoding_field)
		}
	}
	return encoding, err
}

func getFormat(a httpRequestArgs) (Format, errors.Error) {
	var format Format

	format_field, err := a.getString(FORMAT, "JSON")
	if err == nil && format_field != "" {
		format = newFormat(format_field)
		if format == UNDEFINED_FORMAT {
			err = errors.NewServiceErrorUnrecognizedValue(FORMAT, format_field)
		}
	}
	return format, err
}

func getCredentials(a httpRequestArgs,
	hdrCreds *url.Userinfo, auths []string) (datastore.Credentials, errors.Error) {
	var creds datastore.Credentials

	if hdrCreds != nil {
		// Credentials are in the request URL:
		username := hdrCreds.Username()
		password, _ := hdrCreds.Password()
		creds = make(datastore.Credentials)
		creds[username] = password
		return creds, nil
	}
	if len(auths) > 0 {
		// Credentials are in the request header:
		// TODO: implement non-Basic auth (digest, ntlm)
		auth := auths[0]
		if strings.HasPrefix(auth, "Basic ") {
			encoded_creds := strings.Split(auth, " ")[1]
			decoded_creds, err := base64.StdEncoding.DecodeString(encoded_creds)
			if err != nil {
				return creds, errors.NewServiceErrorBadValue(err, CREDS)
			}
			// Authorization header is in format "user:pass"
			// per http://tools.ietf.org/html/rfc1945#section-10.2
			u_details := strings.Split(string(decoded_creds), ":")
			switch len(u_details) {
			case 2:
				creds = make(datastore.Credentials)
				creds[u_details[0]] = u_details[1]
			case 3:
				creds = make(datastore.Credentials)
				// Support usernames like "local:xxx" or "admin:xxx"
				creds[strings.Join(u_details[:2], ":")] = u_details[2]
			default:
				// Authorization header format is incorrect
				return creds, errors.NewServiceErrorBadValue(nil, CREDS)
			}
		}
		return creds, nil
	}
	// Credentials may be in request arguments:
	cred_data, err := a.getCredentials()
	if err == nil && len(cred_data) > 0 {
		creds = make(datastore.Credentials)
		for _, cred := range cred_data {
			user, user_ok := cred["user"]
			pass, pass_ok := cred["pass"]
			if user_ok && pass_ok {
				creds[user] = pass
			} else {
				err = errors.NewServiceErrorMissingValue("user or pass")
				break
			}
		}
	}
	return creds, err
}

// httpRequestArgs is an interface for getting the arguments in a http request
type httpRequestArgs interface {
	getString(string, string) (string, errors.Error)
	getTristate(f string) (value.Tristate, errors.Error)
	getValue(field string) (value.Value, errors.Error)
	getDuration(string) (time.Duration, errors.Error)
	getNamedArgs() (map[string]value.Value, errors.Error)
	getPositionalArgs() (value.Values, errors.Error)
	getStatement() (string, errors.Error)
	getCredentials() ([]map[string]string, errors.Error)
	getScanVector() (timestamp.Vector, errors.Error)
}

// getRequestParams creates a httpRequestArgs implementation,
// depending on the content type in the request
func getRequestParams(req *http.Request) (httpRequestArgs, errors.Error) {

	const (
		URL_CONTENT  = "application/x-www-form-urlencoded"
		JSON_CONTENT = "application/json"
	)
	content_types := req.Header["Content-Type"]
	content_type := URL_CONTENT

	if len(content_types) > 0 {
		content_type = content_types[0]
	}

	if strings.HasPrefix(content_type, URL_CONTENT) {
		return &urlArgs{req: req}, nil
	}

	if strings.HasPrefix(content_type, JSON_CONTENT) {
		return newJsonArgs(req)
	}

	return &urlArgs{req: req}, nil
}

// urlArgs is an implementation of httpRequestArgs that reads
// request arguments from a url-encoded http request
type urlArgs struct {
	req *http.Request
}

func (this *urlArgs) getStatement() (string, errors.Error) {
	statement, err := this.formValue(STATEMENT)
	if err != nil {
		return "", err
	}

	if statement == "" && this.req.Method == "POST" {
		bytes, err := ioutil.ReadAll(this.req.Body)
		if err != nil {
			return "", errors.NewServiceErrorBadValue(err, STATEMENT)
		}

		statement = string(bytes)
	}

	return statement, nil
}

// A named argument is an argument of the form: $<identifier>=json_value
func (this *urlArgs) getNamedArgs() (map[string]value.Value, errors.Error) {
	var args map[string]value.Value

	for name, _ := range this.req.Form {
		if !strings.HasPrefix(name, "$") {
			continue
		}
		arg, err := this.formValue(name)
		if err != nil {
			return args, err
		}
		if len(arg) == 0 {
			//This is an error - there _has_ to be a value for a named argument
			return args, errors.NewServiceErrorMissingValue(fmt.Sprintf("named argument %s", name))
		}
		args = addNamedArg(args, name, value.NewValue([]byte(arg)))
	}
	return args, nil
}

// Positional args are of the form: args=json_list
func (this *urlArgs) getPositionalArgs() (value.Values, errors.Error) {
	var positionalArgs value.Values

	args_field, err := this.formValue(ARGS)
	if err != nil || args_field == "" {
		return positionalArgs, err
	}

	var args []interface{}

	decoder := json.NewDecoder(strings.NewReader(args_field))
	e := decoder.Decode(&args)
	if e != nil {
		return positionalArgs, errors.NewServiceErrorBadValue(e, ARGS)
	}

	positionalArgs = make([]value.Value, len(args))
	// Put each element of args into positionalArgs
	for i, arg := range args {
		positionalArgs[i] = value.NewValue(arg)
	}

	return positionalArgs, nil
}

func (this *urlArgs) getScanVector() (timestamp.Vector, errors.Error) {
	var full_vector_data []*restArg
	var sparse_vector_data map[string]*restArg

	scan_vector_data_field, err := this.formValue(SCAN_VECTOR)

	if err != nil || scan_vector_data_field == "" {
		return nil, err
	}
	decoder := json.NewDecoder(strings.NewReader(scan_vector_data_field))
	e := decoder.Decode(&full_vector_data)
	if e == nil {
		return makeFullVector(full_vector_data)
	}
	decoder = json.NewDecoder(strings.NewReader(scan_vector_data_field))
	e = decoder.Decode(&sparse_vector_data)
	if e != nil {
		return nil, errors.NewServiceErrorBadValue(e, SCAN_VECTOR)
	}
	return makeSparseVector(sparse_vector_data)
}

func (this *urlArgs) getDuration(f string) (time.Duration, errors.Error) {
	var timeout time.Duration

	timeout_field, err := this.formValue(f)
	if err == nil && timeout_field != "" {
		timeout, err = newDuration(timeout_field)
	}
	return timeout, err
}

func (this *urlArgs) getString(f string, dflt string) (string, errors.Error) {
	value := dflt

	value_field, err := this.formValue(f)
	if err == nil && value_field != "" {
		value = value_field
	}
	return value, err
}

func (this *urlArgs) getTristate(f string) (value.Tristate, errors.Error) {
	tristate_value := value.NONE
	value_field, err := this.formValue(f)
	if err != nil {
		return tristate_value, err
	}
	if value_field == "" {
		return tristate_value, nil
	}
	bool_value, e := strconv.ParseBool(value_field)
	if e != nil {
		return tristate_value, errors.NewServiceErrorBadValue(e, f)
	}
	tristate_value = value.ToTristate(bool_value)
	return tristate_value, nil
}

func (this *urlArgs) getCredentials() ([]map[string]string, errors.Error) {
	var creds_data []map[string]string

	creds_field, err := this.formValue(CREDS)
	if err == nil && creds_field != "" {
		decoder := json.NewDecoder(strings.NewReader(creds_field))
		e := decoder.Decode(&creds_data)
		if e != nil {
			err = errors.NewServiceErrorBadValue(e, CREDS)
		}
	}
	return creds_data, err
}

func (this *urlArgs) getValue(field string) (value.Value, errors.Error) {
	var val value.Value
	value_field, err := this.getString(field, "")
	if err == nil && value_field != "" {
		val = value.NewValue([]byte(value_field))
	}
	return val, err
}

func (this *urlArgs) formValue(field string) (string, errors.Error) {
	values := this.req.Form[field]

	switch len(values) {
	case 0:
		return "", nil
	case 1:
		return values[0], nil
	default:
		return "", errors.NewServiceErrorMultipleValues(field)
	}
}

// jsonArgs is an implementation of httpRequestArgs that reads
// request arguments from a json-encoded http request
type jsonArgs struct {
	args map[string]interface{}
	req  *http.Request
}

// create a jsonArgs structure from the given http request.
func newJsonArgs(req *http.Request) (*jsonArgs, errors.Error) {
	var p jsonArgs
	decoder := json.NewDecoder(req.Body)
	err := decoder.Decode(&p.args)
	if err != nil {
		return nil, errors.NewServiceErrorBadValue(err, "JSON request body")
	}
	p.req = req
	return &p, nil
}

func (this *jsonArgs) getStatement() (string, errors.Error) {
	return this.getString(STATEMENT, "")
}

func (this *jsonArgs) getNamedArgs() (map[string]value.Value, errors.Error) {
	var args map[string]value.Value
	for name, arg := range this.args {
		if !strings.HasPrefix(name, "$") {
			continue
		}
		args = addNamedArg(args, name, value.NewValue(arg))
	}
	return args, nil
}

func (this *jsonArgs) getPositionalArgs() (value.Values, errors.Error) {
	var positionalArgs value.Values

	args_field, in_request := this.args[ARGS]
	if !in_request {
		return positionalArgs, nil
	}

	args, type_ok := args_field.([]interface{})
	if !type_ok {
		return positionalArgs, errors.NewServiceErrorTypeMismatch(ARGS, "array")
	}

	positionalArgs = make([]value.Value, len(args))
	// Put each element of args into positionalArgs
	for i, arg := range args {
		positionalArgs[i] = value.NewValue(arg)
	}

	return positionalArgs, nil
}

func (this *jsonArgs) getCredentials() ([]map[string]string, errors.Error) {
	var creds_data []map[string]string

	creds_field, in_request := this.args[CREDS]
	if !in_request {
		return creds_data, nil
	}

	creds_data, type_ok := creds_field.([]map[string]string)
	if !type_ok {
		return creds_data, errors.NewServiceErrorTypeMismatch(CREDS, "array of { user, pass }")
	}

	return creds_data, nil
}

func (this *jsonArgs) getScanVector() (timestamp.Vector, errors.Error) {
	var type_ok bool

	scan_vector_data_field, in_request := this.args[SCAN_VECTOR]
	if !in_request {
		return nil, nil
	}
	full_vector_data, type_ok := scan_vector_data_field.([]interface{})
	if type_ok {
		if len(full_vector_data) != SCAN_VECTOR_SIZE {
			return nil, errors.NewServiceErrorTypeMismatch(SCAN_VECTOR,
				fmt.Sprintf("array of %d entries", SCAN_VECTOR_SIZE))
		}
		entries := make([]timestamp.Entry, len(full_vector_data))
		for index, arg := range full_vector_data {
			nextEntry, err := makeVectorEntry(index, arg)
			if err != nil {
				return nil, err
			}
			entries[index] = nextEntry
		}
	}
	sparse_vector_data, type_ok := scan_vector_data_field.(map[string]interface{})
	if !type_ok {
		return nil, errors.NewServiceErrorTypeMismatch(SCAN_VECTOR, "array or map of { number, string }")
	}
	entries := make([]timestamp.Entry, len(sparse_vector_data))
	i := 0
	for key, arg := range sparse_vector_data {
		index, e := strconv.Atoi(key)
		if e != nil {
			return nil, errors.NewServiceErrorBadValue(e, SCAN_VECTOR)
		}
		nextEntry, err := makeVectorEntry(index, arg)
		if err != nil {
			return nil, err
		}
		entries[i] = nextEntry
		i = i + 1
	}
	return &scanVectorEntries{
		entries: entries,
	}, nil
}

func makeVectorEntry(index int, args interface{}) (*scanVectorEntry, errors.Error) {
	data, is_map := args.(map[string]interface{})
	if !is_map {
		return nil, errors.NewServiceErrorTypeMismatch(SCAN_VECTOR, "array or map of { number, string }")
	}
	seqno, has_seqno := data["seqno"]
	if !has_seqno {
		return nil, errors.NewServiceErrorTypeMismatch(SCAN_VECTOR, "array or map of { number, string }")
	}
	seqno_val, is_number := seqno.(float64)
	if !is_number {
		return nil, errors.NewServiceErrorTypeMismatch(SCAN_VECTOR, "array or map of { number, string }")
	}
	uuid, has_uuid := data["uuid"]
	if !has_uuid {
		return nil, errors.NewServiceErrorTypeMismatch(SCAN_VECTOR, "array or map of { number, string }")
	}
	uuid_val, uuid_ok := uuid.(string)
	if !uuid_ok {
		return nil, errors.NewServiceErrorTypeMismatch(SCAN_VECTOR, "array or map of { number, string }")
	}
	return &scanVectorEntry{
		pos:  uint32(index),
		val:  uint64(seqno_val),
		uuid: uuid_val,
	}, nil
}

func (this *jsonArgs) getDuration(f string) (time.Duration, errors.Error) {
	var timeout time.Duration
	t, err := this.getString(f, "0s")
	if err != nil {
		timeout, err = newDuration(t)
	}
	return timeout, err
}

func (this *jsonArgs) getTristate(f string) (value.Tristate, errors.Error) {
	value_tristate := value.NONE
	value_field, in_request := this.args[f]
	if !in_request {
		return value_tristate, nil
	}

	b, type_ok := value_field.(bool)
	if !type_ok {
		return value_tristate, errors.NewServiceErrorTypeMismatch(f, "boolean")
	}

	value_tristate = value.ToTristate(b)
	return value_tristate, nil
}

// helper function to get a string type argument
func (this *jsonArgs) getString(f string, dflt string) (string, errors.Error) {
	value_field, in_request := this.args[f]
	if !in_request {
		return dflt, nil
	}

	value, type_ok := value_field.(string)
	if !type_ok {
		return value, errors.NewServiceErrorTypeMismatch(f, "string")
	}
	return value, nil
}

func (this *jsonArgs) getValue(f string) (value.Value, errors.Error) {
	var val value.Value
	value_field, in_request := this.args[f]
	if !in_request {
		return val, nil
	}

	val = value.NewValue(value_field)
	return val, nil
}

type Encoding int

const (
	UTF8 Encoding = iota
	UNDEFINED_ENCODING
)

func newEncoding(s string) Encoding {
	switch strings.ToUpper(s) {
	case "UTF-8":
		return UTF8
	default:
		return UNDEFINED_ENCODING
	}
}

func (e Encoding) String() string {
	var s string
	switch e {
	case UTF8:
		s = "UTF-8"
	default:
		s = "UNDEFINED_ENCODING"
	}
	return s
}

type Format int

const (
	JSON Format = iota
	XML
	CSV
	TSV
	UNDEFINED_FORMAT
)

func newFormat(s string) Format {
	switch strings.ToUpper(s) {
	case "JSON":
		return JSON
	case "XML":
		return XML
	case "CSV":
		return CSV
	case "TSV":
		return TSV
	default:
		return UNDEFINED_FORMAT
	}
}

func (f Format) String() string {
	var s string
	switch f {
	case JSON:
		s = "JSON"
	case XML:
		s = "XML"
	case CSV:
		s = "CSV"
	case TSV:
		s = "TSV"
	default:
		s = "UNDEFINED_FORMAT"
	}
	return s
}

type Compression int

const (
	NONE Compression = iota
	ZIP
	RLE
	LZMA
	LZO
	UNDEFINED_COMPRESSION
)

func newCompression(s string) Compression {
	switch strings.ToUpper(s) {
	case "NONE":
		return NONE
	case "ZIP":
		return ZIP
	case "RLE":
		return RLE
	case "LZMA":
		return LZMA
	case "LZO":
		return LZO
	default:
		return UNDEFINED_COMPRESSION
	}
}

func (c Compression) String() string {
	var s string
	switch c {
	case NONE:
		s = "NONE"
	case ZIP:
		s = "ZIP"
	case RLE:
		s = "RLE"
	case LZMA:
		s = "LZMA"
	case LZO:
		s = "LZO"
	default:
		s = "UNDEFINED_COMPRESSION"
	}
	return s
}

// scanVectorEntry implements timestamp.Entry
type scanVectorEntry struct {
	pos  uint32
	val  uint64
	uuid string
}

func (this *scanVectorEntry) Position() uint32 {
	return this.pos
}

func (this *scanVectorEntry) Value() uint64 {
	return this.val
}

func (this *scanVectorEntry) Guard() string {
	return this.uuid
}

// scanVectorEntries implements timestamp.Vector
type scanVectorEntries struct {
	entries []timestamp.Entry
}

func (this *scanVectorEntries) Entries() []timestamp.Entry {
	return this.entries
}

// restArg captures how vector data is passed via REST
type restArg struct {
	Seqno uint64 `json:"seqno"`
	Uuid  string `json:"uuid"`
}

// makeFullVector is used when the request includes all entries
func makeFullVector(args []*restArg) (*scanVectorEntries, errors.Error) {
	if len(args) != SCAN_VECTOR_SIZE {
		return nil, errors.NewServiceErrorTypeMismatch(SCAN_VECTOR,
			fmt.Sprintf("array of %d entries", SCAN_VECTOR_SIZE))
	}
	entries := make([]timestamp.Entry, len(args))
	for i, arg := range args {
		entries[i] = &scanVectorEntry{
			pos:  uint32(i),
			val:  arg.Seqno,
			uuid: arg.Uuid,
		}
	}
	return &scanVectorEntries{
		entries: entries,
	}, nil
}

// makeSparseVector is used when the request contains a sparse entry arg
func makeSparseVector(args map[string]*restArg) (*scanVectorEntries, errors.Error) {
	entries := make([]timestamp.Entry, len(args))
	i := 0
	for key, arg := range args {
		index, err := strconv.Atoi(key)
		if err != nil {
			return nil, errors.NewServiceErrorBadValue(err, SCAN_VECTOR)
		}
		entries[i] = &scanVectorEntry{
			pos:  uint32(index),
			val:  arg.Seqno,
			uuid: arg.Uuid,
		}
		i = i + 1
	}
	return &scanVectorEntries{
		entries: entries,
	}, nil
}

const SCAN_VECTOR_SIZE = 1024

type scanConfigImpl struct {
	scan_level  server.ScanConsistency
	scan_wait   time.Duration
	scan_vector timestamp.Vector
}

func (this *scanConfigImpl) ScanConsistency() datastore.ScanConsistency {
	switch this.scan_level {
	case server.NOT_BOUNDED:
		return datastore.UNBOUNDED
	case server.REQUEST_PLUS, server.STATEMENT_PLUS:
		return datastore.SCAN_PLUS
	case server.AT_PLUS:
		return datastore.AT_PLUS
	default:
		return datastore.UNBOUNDED
	}
}

func (this *scanConfigImpl) ScanWait() time.Duration {
	return this.scan_wait
}

func (this *scanConfigImpl) ScanVector() timestamp.Vector {
	return this.scan_vector
}

func newScanConsistency(s string) server.ScanConsistency {
	switch strings.ToUpper(s) {
	case "NOT_BOUNDED":
		return server.NOT_BOUNDED
	case "REQUEST_PLUS":
		return server.REQUEST_PLUS
	case "STATEMENT_PLUS":
		return server.STATEMENT_PLUS
	case "AT_PLUS":
		return server.AT_PLUS
	default:
		return server.UNDEFINED_CONSISTENCY
	}
}

// addNamedArgs is used by getNamedArgs implementations to add a named argument
func addNamedArg(args map[string]value.Value, name string, arg value.Value) map[string]value.Value {
	if args == nil {
		args = make(map[string]value.Value)
	}
	// The '$' is trimmed from the argument name when added to args:
	args[strings.TrimPrefix(name, "$")] = arg
	return args
}

// helper function to create a time.Duration instance from a given string.
// There must be a unit - valid units are "ns", "us", "ms", "s", "m", "h"
func newDuration(s string) (duration time.Duration, err errors.Error) {
	// Error if given string has no unit
	last_char := s[len(s)-1]
	if last_char != 's' && last_char != 'm' && last_char != 'h' {
		err = errors.NewServiceErrorBadValue(nil,
			fmt.Sprintf("duration value %s: missing or incorrect unit "+
				"(valid units: ns, us, ms, s, m, h)", s))
	}
	if err == nil {
		d, e := time.ParseDuration(s)
		if e != nil {
			err = errors.NewServiceErrorBadValue(e, "duration")
		} else {
			duration = d
		}
	}
	return
}
