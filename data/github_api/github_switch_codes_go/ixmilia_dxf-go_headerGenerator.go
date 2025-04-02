// Repository: ixmilia/dxf-go
// File: generator/headerGenerator.go

package main

import (
	"encoding/xml"
	"fmt"
	"io"
	"os"
	"strings"
)

type xmlHeader struct {
	XMLName   xml.Name            `xml:"Header"`
	Variables []xmlHeaderVariable `xml:"Variable"`
}

type xmlHeaderVariable struct {
	XMLName          xml.Name                `xml:"Variable"`
	Name             string                  `xml:"Name,attr"`
	Code             int                     `xml:"Code,attr"`
	Type             string                  `xml:"Type,attr"`
	FieldName        string                  `xml:"Field,attr"`
	DefaultValue     string                  `xml:"DefaultValue,attr"`
	ReadConverter    string                  `xml:"ReadConverter,attr"`
	WriteConverter   string                  `xml:"WriteConverter,attr"`
	MinVersion       string                  `xml:"MinVersion,attr"`
	MaxVersion       string                  `xml:"MaxVersion,attr"`
	SuppressWriting  bool                    `xml:"SuppressWriting,attr"`
	DontWriteDefault bool                    `xml:"DontWriteDefault,attr"`
	Comment          string                  `xml:"Comment,attr"`
	Flags            []xmlHeaderVariableFlag `xml:"Flag"`
}

type xmlHeaderVariableFlag struct {
	XMLName xml.Name `xml:"Flag"`
	Name    string   `xml:"Name,attr"`
	Mask    int      `xml:"Mask,attr"`
	Comment string   `xml:"Comment,attr"`
}

func generateHeader() {
	specPath := "spec/HeaderSpec.xml"
	file, err := os.Open(specPath)
	check(err)

	defer file.Close()

	variables, err := readHeader(file)
	check(err)

	var builder strings.Builder
	builder.WriteString("// Code generated at build time; DO NOT EDIT.\n")
	builder.WriteString("\n")
	builder.WriteString("package dxf\n")
	builder.WriteString("\n")
	builder.WriteString("import (\n")
	builder.WriteString("	\"errors\"\n")
	builder.WriteString("	\"math\"\n")
	builder.WriteString("	\"time\"\n")
	builder.WriteString("\n")
	builder.WriteString("	\"github.com/google/uuid\"\n")
	builder.WriteString(")\n")
	builder.WriteString("\n")

	// definition
	builder.WriteString("// Header contains the values common to an AutoCAD DXF drawing.\n")
	builder.WriteString("type Header struct {\n")
	seenVariables := make(map[string]bool)
	for _, variable := range variables {
		if !seenVariables[variable.FieldName] {
			seenVariables[variable.FieldName] = true
			comment := generateComment(fmt.Sprintf("The $%s header variable.  %s", variable.Name, variable.Comment), variable.MinVersion, variable.MaxVersion)
			builder.WriteString(fmt.Sprintf("	// %s\n", comment))
			builder.WriteString(fmt.Sprintf("	%s %s\n", variable.FieldName, variable.Type))
		}
	}
	builder.WriteString("}\n")
	builder.WriteString("\n")

	// constructor
	builder.WriteString("func NewHeader() *Header {\n")
	builder.WriteString("	return &Header{\n")
	seenVariables = make(map[string]bool)
	for _, variable := range variables {
		if !seenVariables[variable.FieldName] {
			seenVariables[variable.FieldName] = true
			builder.WriteString(fmt.Sprintf("		%s: %s,\n", variable.FieldName, variable.DefaultValue))
		}
	}
	builder.WriteString("	}\n")
	builder.WriteString("}\n")
	builder.WriteString("\n")

	// flags
	for _, variable := range variables {
		for _, flag := range variable.Flags {
			comment := generateComment(fmt.Sprintf("%s status flag.", flag.Name), variable.MinVersion, variable.MaxVersion)
			builder.WriteString(fmt.Sprintf("// %s\n", comment))
			builder.WriteString(fmt.Sprintf("func (h *Header) %s() (val bool) {\n", flag.Name))
			builder.WriteString(fmt.Sprintf("	return h.%s & %d != 0\n", variable.FieldName, flag.Mask))
			builder.WriteString("}\n")
			builder.WriteString("\n")

			builder.WriteString(fmt.Sprintf("// %s\n", comment))
			builder.WriteString(fmt.Sprintf("func (h *Header) Set%s(val bool) {\n", flag.Name))
			builder.WriteString("	if val {\n")
			builder.WriteString(fmt.Sprintf("		h.%s = h.%s | %d\n", variable.FieldName, variable.FieldName, flag.Mask))
			builder.WriteString("	} else {\n")
			builder.WriteString(fmt.Sprintf("		h.%s = h.%s & ^%d\n", variable.FieldName, variable.FieldName, flag.Mask))
			builder.WriteString("	}\n")
			builder.WriteString("}\n")
			builder.WriteString("\n")
		}
	}

	// writeHeaderSection()
	builder.WriteString("func (h Header) writeHeaderSection(writer codePairWriter) error {\n")
	builder.WriteString("	pairs := make([]CodePair, 0)\n")
	for _, variable := range variables {
		if variable.SuppressWriting {
			continue
		}
		builder.WriteString("\n")
		builder.WriteString(fmt.Sprintf("	// $%s\n", variable.Name))
		var predicates []string
		if len(variable.MinVersion) > 0 {
			predicates = append(predicates, fmt.Sprintf("h.Version >= %s", variable.MinVersion))
		}
		if len(variable.MaxVersion) > 0 {
			predicates = append(predicates, fmt.Sprintf("h.Version <= %s", variable.MaxVersion))
		}
		if variable.DontWriteDefault {
			predicates = append(predicates, fmt.Sprintf("h.%s != %s", variable.FieldName, variable.DefaultValue))
		}
		indention := ""
		if len(predicates) > 0 {
			indention = "	"
			builder.WriteString(fmt.Sprintf("	if %s {\n", strings.Join(predicates, " && ")))
		}
		builder.WriteString(fmt.Sprintf("%s	pairs = append(pairs, NewStringCodePair(9, \"$%s\"))\n", indention, variable.Name))
		value := fmt.Sprintf("h.%s", variable.FieldName)
		if len(variable.WriteConverter) > 0 {
			value = strings.Replace(variable.WriteConverter, "%v", value, -1)
		}

		// write the value
		if variable.Code < 0 {
			// either a Point or Vector
			for code := 10; code <= variable.Code*-10; code += 10 {
				component := 'X' + (code-10)/10
				builder.WriteString(fmt.Sprintf("%s	pairs = append(pairs, NewDoubleCodePair(%d, h.%s.%c))\n", indention, code, variable.FieldName, component))
			}
		} else {
			typeName := codeTypeName(variable.Code)
			builder.WriteString(fmt.Sprintf("%s	pairs = append(pairs, New%sCodePair(%d, %s))\n", indention, typeName, variable.Code, value))
		}

		if len(predicates) > 0 {
			builder.WriteString(fmt.Sprintf("	}\n"))
		}
	}
	builder.WriteString("\n")
	builder.WriteString("	err := writeSectionStart(writer, \"HEADER\")\n")
	builder.WriteString("	if err != nil {\n")
	builder.WriteString("		return err\n")
	builder.WriteString("	}\n")
	builder.WriteString("	for _, pair := range pairs {\n")
	builder.WriteString("		err = writer.writeCodePair(pair)\n")
	builder.WriteString("		if err != nil {\n")
	builder.WriteString("			return err\n")
	builder.WriteString("		}\n")
	builder.WriteString("	}\n")
	builder.WriteString("	err = writeSectionEnd(writer)\n")
	builder.WriteString("	if err != nil {\n")
	builder.WriteString("		return err\n")
	builder.WriteString("	}\n")
	builder.WriteString("	return nil\n")
	builder.WriteString("}\n")
	builder.WriteString("\n")

	// readHeader()
	builder.WriteString("func readHeader(nextPair CodePair, reader codePairReader) (Header, CodePair, error) {\n")
	builder.WriteString("	header := *NewHeader()\n")
	builder.WriteString("	var err error\n")
	builder.WriteString("	var variableName string\n")
	builder.WriteString("	for nextPair.Code != 0 {\n")

	// look for 9/$VARNAME
	builder.WriteString("		if nextPair.Code == 9 {\n")
	builder.WriteString("			variableName = nextPair.Value.(StringCodePairValue).Value\n")
	builder.WriteString("		} else {\n")
	builder.WriteString("			switch variableName {\n")
	seenVariables = make(map[string]bool)
	for _, variable := range variables {
		if !seenVariables[variable.FieldName] {
			seenVariables[variable.FieldName] = true
			builder.WriteString(fmt.Sprintf("			case \"$%s\":\n", variable.Name))
			// read the value
			if variable.Code < 0 {
				// either a Point or a Vector
				builder.WriteString("				switch nextPair.Code {\n")
				for code := 10; code <= variable.Code*-10; code += 10 {
					component := 'X' + (code-10)/10
					builder.WriteString(fmt.Sprintf("				case %d:\n", code))
					builder.WriteString(fmt.Sprintf("					header.%s.%c = nextPair.Value.(DoubleCodePairValue).Value\n", variable.FieldName, component))
				}
				builder.WriteString("				}\n")
			} else {
				// validate all possible codes; there are some duplicates
				duplicateVariables := make([]xmlHeaderVariable, 0)
				for _, v := range variables {
					if v.Name == variable.Name {
						duplicateVariables = append(duplicateVariables, v)
					}
				}

				if len(duplicateVariables) > 1 {
					builder.WriteString("				switch nextPair.Code {\n")
					allowedCodes := make([]string, 0)
					for _, v := range duplicateVariables {
						allowedCodes = append(allowedCodes, fmt.Sprintf("%d", v.Code))
						builder.WriteString(fmt.Sprintf("				case %d:\n", v.Code))
						builder.WriteString(fmt.Sprintf("					header.%s = %s\n", v.FieldName, generateReadFunction(v)))
					}
					builder.WriteString("				default:\n")
					builder.WriteString(fmt.Sprintf("					return header, nextPair, errors.New(\"expected codes %s\")\n", strings.Join(allowedCodes, ", ")))
					builder.WriteString("				}\n")
				} else {
					builder.WriteString(fmt.Sprintf("				if nextPair.Code != %d {\n", variable.Code))
					builder.WriteString(fmt.Sprintf("					return header, nextPair, errors.New(\"expected code %d\")\n", variable.Code))
					builder.WriteString("				}\n")
					builder.WriteString(fmt.Sprintf("				header.%s = %s\n", variable.FieldName, generateReadFunction(variable)))
				}
			}

			if variable.Name == "ACADVER" {
				builder.WriteString(fmt.Sprintf("				if header.%s >= R2007 {\n", variable.FieldName))
				builder.WriteString("					reader.setUtf8Reader()\n")
				builder.WriteString("				}\n")
			}
		}
	}
	builder.WriteString("			default:\n")
	builder.WriteString("				// ignore unsupported header variable\n")
	builder.WriteString("			}\n")
	builder.WriteString("		}\n")
	builder.WriteString("\n")
	builder.WriteString("		nextPair, err = reader.readCodePair()\n")
	builder.WriteString("		if err != nil {\n")
	builder.WriteString("			return header, nextPair, err\n")
	builder.WriteString("		}\n")
	builder.WriteString("	}\n")
	builder.WriteString("\n")
	builder.WriteString("	return header, nextPair, nil\n")
	builder.WriteString("}\n")

	writeFile("header.generated.go", builder)
}

func generateReadFunction(variable xmlHeaderVariable) string {
	readValue := fmt.Sprintf("nextPair.Value.(%sCodePairValue).Value", codeTypeName(variable.Code))
	if len(variable.ReadConverter) > 0 {
		readValue = strings.Replace(variable.ReadConverter, "%v", readValue, -1)
	}

	return readValue
}

func readHeader(reader io.Reader) ([]xmlHeaderVariable, error) {
	var header xmlHeader
	if err := xml.NewDecoder(reader).Decode(&header); err != nil {
		return nil, err
	}

	return header.Variables, nil
}
