// Repository: fluent-qa/Go-Steps
// File: go_examples/switch.go

package main

import (
	"time"
	"fmt"
)

func main(){
	i:=2
	fmt.Println("Write",i," as ")
	switch i {
	case 1:
		fmt.Println("one")
	case 2:
		fmt.Println("two")
	case 3:
		fmt.Println("three")
	}

	switch time.Now().Weekday(){
	case time.Saturday,time.Sunday:
		fmt.Println("It is the weekend")
	default:
		fmt.Println("It is weekday")
	}

	t:=time.Now()
	switch{
	case t.Hour()<12:
		fmt.Println("It is before noon")
	default:
		fmt.Println("It is after noon")
	}

	whatAmI:=func(i interface{}){
		switch t:=i.(type){
		case bool:
			fmt.Println("I am a bool")
		case int:
			fmt.Println("I am an int")
		default:
			fmt.Printf("Don't know type %T\n",t)
		}
	}
	whatAmI(true)
	whatAmI(1)
	whatAmI("hey")
}