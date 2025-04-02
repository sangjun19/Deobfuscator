// Repository: rai-project/dlperf
// File: pkg/onnx/utils.go

package onnx

import (
	"encoding/binary"
	"fmt"
	"io/ioutil"
	"path"
	"path/filepath"
	"reflect"
	"strings"

	"github.com/Unknwon/com"
	"github.com/pkg/errors"
	dlperf "github.com/rai-project/dlperf/pkg"
	"github.com/rai-project/onnx"
	"github.com/spf13/cast"
)

func getNodeAttributeFromName(node *onnx.NodeProto, attrName string) *onnx.AttributeProto {
	for _, attr := range node.GetAttribute() {
		if attr.GetName() == attrName {
			return attr
		}
	}

	return nil
}

func getTensorProtoDimensions(tensor *onnx.TensorProto) dlperf.Shape {
	var ret dlperf.Shape

	isInt32 := (tensor.DataType == int32(onnx.TensorProto_INT32) || tensor.DataType == int32(onnx.TensorProto_UINT32))
	if isInt32 && len(tensor.GetInt32Data()) > 0 {
		return toInt64Slice(tensor.GetInt32Data())
	}
	if isInt32 && len(tensor.GetRawData()) > 0 && len(tensor.Dims) > 0 {
		dim := tensor.Dims[0]
		rawdata := tensor.GetRawData()
		for ii := int64(0); ii < dim; ii++ {
			val := int64(binary.LittleEndian.Uint32(rawdata[ii*4 : (ii+1)*4]))
			ret = append(ret, val)
		}
		return ret
	}

	isInt64 := (tensor.DataType == int32(onnx.TensorProto_INT64) || tensor.DataType == int32(onnx.TensorProto_UINT64))
	if isInt64 && len(tensor.GetInt64Data()) > 0 {
		return tensor.GetInt64Data()
	}
	if isInt64 && len(tensor.GetRawData()) > 0 && len(tensor.Dims) > 0 {
		dim := tensor.Dims[0]
		rawdata := tensor.GetRawData()
		for ii := int64(0); ii < dim; ii++ {
			val := int64(binary.LittleEndian.Uint64(rawdata[ii*8 : (ii+1)*8]))
			ret = append(ret, val)
		}
		return ret
	}

	return tensor.Dims
}

func getValueInfoDimensions(valueInfo *onnx.ValueInfoProto) dlperf.Shape {
	var ret dlperf.Shape
	for _, dim := range valueInfo.GetType().GetTensorType().GetShape().GetDim() {
		ret = append(ret, dim.GetDimValue())
	}
	return ret
}

// toIntSliceE casts an interface to a []int32 type.
func toInt32SliceE(i interface{}) ([]int32, error) {
	if i == nil {
		return []int32{}, fmt.Errorf("unable to cast %#v of type %T to []int32", i, i)
	}

	switch v := i.(type) {
	case []int32:
		return v, nil
	}

	kind := reflect.TypeOf(i).Kind()
	switch kind {
	case reflect.Slice, reflect.Array:
		s := reflect.ValueOf(i)
		a := make([]int32, s.Len())
		for j := 0; j < s.Len(); j++ {
			val, err := cast.ToInt32E(s.Index(j).Interface())
			if err != nil {
				return []int32{}, fmt.Errorf("unable to cast %#v of type %T to []int32", i, i)
			}
			a[j] = val
		}
		return a, nil
	default:
		return []int32{}, fmt.Errorf("unable to cast %#v of type %T to []int32", i, i)
	}
}

func toInt32Slice(i interface{}) []int32 {
	v, _ := toInt32SliceE(i)
	return v
}

// toIntSliceE casts an interface to a []int64 type.
func toInt64SliceE(i interface{}) ([]int64, error) {
	if i == nil {
		return []int64{}, fmt.Errorf("unable to cast %#v of type %T to []int64", i, i)
	}

	switch v := i.(type) {
	case []int64:
		return v, nil
	}

	kind := reflect.TypeOf(i).Kind()
	switch kind {
	case reflect.Slice, reflect.Array:
		s := reflect.ValueOf(i)
		a := make([]int64, s.Len())
		for j := 0; j < s.Len(); j++ {
			val, err := cast.ToInt64E(s.Index(j).Interface())
			if err != nil {
				return []int64{}, fmt.Errorf("unable to cast %#v of type %T to []int64", i, i)
			}
			a[j] = val
		}
		return a, nil
	default:
		return []int64{}, fmt.Errorf("unable to cast %#v of type %T to []int64", i, i)
	}
}

func toInt64Slice(i interface{}) []int64 {
	v, _ := toInt64SliceE(i)
	return v
}

func getOutputShapes(layers dlperf.Layers) []dlperf.Shape {
	outputShapes := []dlperf.Shape{}
	for _, layer := range layers {
		os := layer.OutputShapes()
		if len(os) == 0 {
			continue
		}
		outputShapes = append(outputShapes, os[0])
	}

	return outputShapes
}

func getModelName(modelPath string) string {
	return iGetModelName(modelPath, "")
}

func findModelNameFile(dir string, level int) (string, error) {
	if level < 0 {
		return "", errors.New("unable to find model_name file")
	}
	pth := filepath.Join(dir, "model_name")
	if com.IsFile(pth) {
		bts, err := ioutil.ReadFile(pth)
		if err != nil {
			return "", err
		}
		return string(bts), err
	}
	return findModelNameFile(filepath.Dir(dir), level-1)
}

func iGetModelName(modelPath, suffix string) string {
	name, err := findModelNameFile(filepath.Dir(modelPath), 4 /* levels */)
	if err == nil {
		return removeSpace(name)
	}
	name = strings.TrimSuffix(filepath.Base(modelPath), ".onnx")
	if name != "model" && name != "model_inferred" {
		return removeSpace(name + suffix)
	}
	if suffix == "" && name == "model_inferred" {
		suffix = "_inferred"
	}
	return iGetModelName(path.Dir(modelPath), suffix)
}

func removeSpace(s string) string {
	s = strings.TrimSuffix(s, "\n")
	s = strings.TrimSpace(s)
	return s
}
