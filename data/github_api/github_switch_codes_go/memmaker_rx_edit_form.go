// Repository: memmaker/rx_edit
// File: stk/form.go

package stk

import (
	"github.com/memmaker/go/recfile"
	. "modernc.org/tk9.0"
	"strconv"
	"strings"
)

type GUIFieldType uint8

const (
	GUIFieldText GUIFieldType = iota
	GUIFieldMultilineText
	GUIFieldInteger
	GUIFieldFloat
	GUIFieldBoolean
	GUIFieldChoice
	GUIFieldReference
)

type FormField struct {
	FieldName         string
	PrefilledValues   []string
	Options           []Option
	FieldType         GUIFieldType
	IsList            bool
	Label             string
	ReferencedSchemas []string
}

func (f FormField) PrefilledValue() string {
	if len(f.PrefilledValues) > 0 {
		return f.PrefilledValues[0]
	}
	return ""
}

type Option struct {
	Label string
	Value string
}

func AskForString(parent *Window, prompt string, onSubmission func(value string)) {
	OpenForm(parent, [][]FormField{{{
		FieldName: "value",
		Label:     prompt,
		FieldType: GUIFieldText,
	}}}, nil, func(record recfile.Record) {
		onSubmission(record[0].Value)
	})
}
func OpenForm(parent *Window, formDefPerColumn [][]FormField, listRecordIds func(recordType string) []string, onSubmission func(recfile.Record)) {
	if parent == nil {
		parent = App
	}
	StyleConfigure("invalid.TEntry", Foreground("red"))
	StyleConfigure("valid.TEntry", Foreground("white"))

	formColumnCount := len(formDefPerColumn)
	columnFrames := make([]*TFrameWidget, formColumnCount)
	listWidgets := make(map[string]*WidgetList)
	widgets := make(map[string]Widget)

	for columnIndex, formDef := range formDefPerColumn {
		formFrame := parent.TFrame()

		for i, field := range formDef {
			labelStick := "e"
			if field.FieldType == GUIFieldMultilineText || field.IsList {
				labelStick = "ne"
			}

			fieldRow := Row(i)
			fieldValueColumn := Column(1)

			Grid(formFrame.TLabel(Txt(field.Label)), fieldRow, Column(0), Sticky(labelStick), Padx(4))

			if field.IsList {
				// first occurrence, place a list frame instead of the field
				listFrame := NewWidgetList(formFrame.Window, field, func(parent *Window, fieldDef FormField) []Widget {
					newWidget, sticky := CreateFormField(parent, fieldDef, listRecordIds)
					Grid(newWidget, Sticky(sticky))
					return []Widget{newWidget}
				})
				Grid(listFrame.container, fieldRow, Column(1), Sticky("we"))
				listWidgets[field.FieldName] = listFrame
			} else {
				formFieldWidget, sticky := CreateFormField(formFrame.Window, field, listRecordIds)
				widgets[field.FieldName] = formFieldWidget
				Grid(formFieldWidget, fieldRow, fieldValueColumn, Sticky(sticky))
			}

			if field.FieldType == GUIFieldMultilineText {
				GridRowConfigure(formFrame, i, Weight(1))
			} else {
				GridRowConfigure(formFrame, i, Weight(0))
			}
		}

		GridColumnConfigure(formFrame, 0, Weight(0))
		GridColumnConfigure(formFrame, 1, Weight(1))

		Grid(formFrame, Row(0), Column(columnIndex), Sticky("wens"))

		columnFrames[columnIndex] = formFrame
		GridColumnConfigure(parent, columnIndex, Weight(1))
	}

	GridRowConfigure(parent, 0, Weight(1))
	GridRowConfigure(parent, 1, Weight(0))

	buttonFrame := parent.TFrame()
	Grid(buttonFrame, Row(1), Column(0), Sticky("we"), Columnspan(formColumnCount))

	cancelButton := buttonFrame.TButton(Txt("Cancel"), Command(func() { Destroy(parent) }))
	submitButton := buttonFrame.TButton(Txt("Submit"), Command(func() {
		currentValues := make(recfile.Record, 0)
		listFieldsHandled := make(map[string]bool)
		validationErrors := false
		for _, formDef := range formDefPerColumn {
			for _, field := range formDef {
				if field.IsList {
					if _, ok := listFieldsHandled[field.FieldName]; ok {
						continue
					}
					listWidget := listWidgets[field.FieldName]
					values := listWidget.Values(func(widgets []Widget) string {
						value, _ := StringFromWidget(widgets[0], field)
						return value
					})
					for _, value := range values {
						if value == "" {
							continue
						}
						currentValues = append(currentValues, recfile.Field{Name: field.FieldName, Value: value})
					}
					listFieldsHandled[field.FieldName] = true
					continue
				}
				value, isValid := StringFromWidget(widgets[field.FieldName], field)
				if !isValid {
					validationErrors = true
				}
				if value == "" {
					continue
				}
				currentValues = append(currentValues, recfile.Field{Name: field.FieldName, Value: value})
			}
		}
		if validationErrors {
			return
		}

		onSubmission(currentValues)
		Destroy(parent)
	}))

	Grid(cancelButton, Row(0), Column(0))
	Grid(submitButton, Row(0), Column(1))
	GridColumnConfigure(buttonFrame, 0, Weight(1))
	GridColumnConfigure(buttonFrame, 1, Weight(1))
}

func StringFromWidget(widget Widget, field FormField) (string, bool) {
	switch typedWidget := widget.(type) {
	case *TEntryWidget:
		widgetValue := typedWidget.Textvariable()
		if !valid(field, widgetValue) {
			typedWidget.Configure(Style("invalid.TEntry"))
			return widgetValue, false
		} else {
			typedWidget.Configure(Style("valid.TEntry"))
			return widgetValue, true
		}
	case *TextWidget:
		value := strings.TrimSpace(strings.Join(typedWidget.Get("1.0", "end"), ""))
		return value, true
	case *TCheckbuttonWidget:
		return typedWidget.Variable(), true
	case *TComboboxWidget:
		dropDownIndex, _ := strconv.Atoi(typedWidget.Current(nil))
		realIndex := dropDownIndex - 1
		if realIndex < 0 {
			return "", true
		}
		value := field.Options[realIndex].Value
		return value, true
	}
	return "", false
}

func CreateFormField(parentOfField *Window, field FormField, listRecordIds func(recordType string) []string) (Widget, string) {
	var tField Widget
	switch field.FieldType {
	case GUIFieldText:
		tField = parentOfField.TEntry(Textvariable(field.PrefilledValue()))
	case GUIFieldMultilineText:
		tFieldMulti := parentOfField.Text(Wrap("word"), Height(3), Width(26))
		tFieldMulti.Insert("end", field.PrefilledValue())
		tField = tFieldMulti
	case GUIFieldInteger:
		tField = parentOfField.TEntry(Textvariable(field.PrefilledValue()))
	case GUIFieldFloat:
		tField = parentOfField.TEntry(Textvariable(field.PrefilledValue()))
	case GUIFieldBoolean:
		checkButtonOptions := []Opt{Offvalue("false"), Onvalue("true")}
		if field.PrefilledValue() != "" {
			checkButtonOptions = append(checkButtonOptions, Variable(field.PrefilledValue() == "true"))
		}
		tField = parentOfField.TCheckbutton(checkButtonOptions...)
	case GUIFieldChoice:
		labels := []string{"<None>"}
		var preselected int
		for optIndex, option := range field.Options {
			label := option.Label
			if label == "" {
				label = option.Value
			}
			labels = append(labels, "{"+strings.ReplaceAll(label, " ", "\u00A0")+"}")
			if strings.ToLower(option.Value) == strings.ToLower(field.PrefilledValue()) {
				preselected = optIndex + 1
			}
		}
		tFieldCombo := parentOfField.TCombobox(Values(strings.Join(labels, " ")), State("readonly"))
		tFieldCombo.Current(preselected)
		tField = tFieldCombo
	case GUIFieldReference:
		frame := parentOfField.TFrame()
		boundVariable := field.PrefilledValue()
		textEntry := frame.TEntry(Textvariable(boundVariable))
		referenceButton := frame.TButton(Txt("..."), Command(func() {
			// Open a new window to select a record from the referenced schema
			// and fill the entry with the key field of the selected record
			newSelectorWindow := App.Toplevel()
			newSelectorWindow.WmTitle("Select a record")
			if len(field.ReferencedSchemas) == 1 {
				OpenList(newSelectorWindow.Window, listRecordIds(field.ReferencedSchemas[0]), func(index int, recordId string) {
					textEntry.Configure(Textvariable(recordId))
					Destroy(newSelectorWindow)
				})
			} else if len(field.ReferencedSchemas) > 1 {
				childNodes := make(map[string][]string)
				for _, schema := range field.ReferencedSchemas {
					childNodes[schema] = listRecordIds(schema)
				}
				OpenTree(newSelectorWindow.Window, field.ReferencedSchemas, childNodes, func(tree *TTreeviewWidget, parentIndex int, parent string, childIndex int, child string) {
					textEntry.Configure(Textvariable(child))
					Destroy(newSelectorWindow)
				})
			}
		}))
		Grid(textEntry, Row(0), Column(0), Sticky("we"))
		Grid(referenceButton, Row(0), Column(1), Sticky("w"))
		tField = frame
	}

	sticky := "we"
	if field.FieldType == GUIFieldMultilineText {
		sticky = "wens"
	} else if field.FieldType == GUIFieldBoolean {
		sticky = "w"
	}
	return tField, sticky
}

func valid(field FormField, value string) bool {
	if value == "" {
		return true
	}
	switch field.FieldType {
	case GUIFieldInteger:
		_, err := strconv.Atoi(value)
		return err == nil
	case GUIFieldFloat:
		_, err := strconv.ParseFloat(value, 64)
		return err == nil
	}
	return true
}
