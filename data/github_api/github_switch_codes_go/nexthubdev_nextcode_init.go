// Repository: nexthubdev/nextcode
// File: cmd/internal/mod/init.go

/*
 * Copyright (c) NeXTHub Corporation. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 */

package mod

import (
	"log"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/goplus/gop/cmd/internal/base"
	"github.com/goplus/gop/env"
	"github.com/goplus/mod/gopmod"
	"github.com/goplus/mod/modload"
)

// gop mod init
var cmdInit = &base.Command{
	UsageLine: "gop mod init [-llgo -tinygo] module-path",
	Short:     "initialize new module in current directory",
}

var (
	flagInit   = &cmdInit.Flag
	flagLLGo   = flagInit.Bool("llgo", false, "use llgo as the compiler")
	flagTinyGo = flagInit.Bool("tinygo", false, "use tinygo as the compiler")
)

func init() {
	cmdInit.Run = runInit
}

func runInit(cmd *base.Command, args []string) {
	err := flagInit.Parse(args)
	if err != nil {
		log.Fatalln("parse input arguments failed:", err)
	}
	args = flagInit.Args()
	switch len(args) {
	case 0:
		fatal(`Example usage:
	'gop mod init example.com/m' to initialize a v0 or v1 module
	'gop mod init example.com/m/v2' to initialize a v2 module

Run 'gop help mod init' for more information.`)
	case 1:
	default:
		fatal("gop mod init: too many arguments")
	}

	modPath := args[0]
	mod, err := modload.Create(".", modPath, goMainVer(), env.MainVersion)
	check(err)

	if *flagLLGo {
		mod.AddCompiler("llgo", "1.0")
		mod.AddRequire("github.com/goplus/llgo", llgoVer(), false)
	} else if *flagTinyGo {
		mod.AddCompiler("tinygo", "0.32")
	}

	err = mod.Save()
	check(err)
}

func goMainVer() string {
	ver := strings.TrimPrefix(runtime.Version(), "go")
	if pos := strings.Index(ver, "."); pos > 0 {
		pos++
		if pos2 := strings.Index(ver[pos:], "."); pos2 > 0 {
			ver = ver[:pos+pos2]
		}
	}
	return ver
}

func llgoVer() string {
	if modGop, e1 := gopmod.LoadFrom(filepath.Join(env.GOPROOT(), "go.mod"), ""); e1 == nil {
		if pkg, e2 := modGop.Lookup("github.com/goplus/llgo"); e2 == nil {
			return pkg.Real.Version
		}
	}
	return "v0.9.0"
}
