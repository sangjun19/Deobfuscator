// Repository: Plan9-Archive/gproc
// File: src/gproc/mexec.go

/*
 * gproc, a Go reimplementation of the LANL version of bproc and the LANL XCPU software. 
 * 
 * This software is released under the GNU Lesser General Public License,
 * version 2, incorporated herein by reference. 
 *
 * Copyright (2010) Sandia Corporation. Under the terms of Contract 
 * DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains 
 * certain rights in this software.
 */

/*
 * The functions in this file are meant to be called when you give an "e" command,
 * to execute a program, on the command line. They pack up the necessary files and
 * send them to the master for distribution
 */

package main

import (
	"bitbucket.org/floren/gproc/src/ldd"
	"errors"
	"fmt"
	"log"
	"os"
	"path"
	"path/filepath"
	"strings"
	"syscall"
)

/*
 * This function is called when you give gproc an "e" argument, in order to run a specified
 * command on the selected nodes
 */
func startExecution(masterAddr, fam, ioProxyPort, slaveNodes string, cmd []string) {
	log.SetPrefix("mexec " + *prefix + ": ")
	/* make sure there is someone to talk to, and get the vital data */
	client, err := Dial("unix", "", masterAddr)
	if err != nil {
		log_error("startExecution: dialing: ", fam, " ", masterAddr, " ", err)
	}
	r := NewRpcClientServer(client, *binRoot)

	/* master sends us vital data */
	var vitalData vitalData
	if r.Recv("vitalData", &vitalData) != nil {
		log_error("Can't get vital data from master")
	}
	pv := newPackVisitor()
	cwd, _ := os.Getwd()
	/* make sure our cwd ends up in the list of things to take along ...  but only take the dir*/
	filepath.Walk(cwd+"/.", walkFunc(pv, nil))
	if len(*filesToTakeAlong) > 0 {
		files := strings.SplitN(*filesToTakeAlong, ",", -1)
		for _, f := range files {
			rootedpath := f
			if f[0] != '/' {
				rootedpath = cwd + "/" + f
			}
			filepath.Walk(rootedpath, walkFunc(pv, nil))
		}
	}
	rawFiles, _ := ldd.Lddroot(cmd[0], *root, *libs)
	log_info("LDD say rawFiles ", rawFiles, "cmds ", cmd, "root ", *root, " libs ", *libs)

	/* now filter out the files we will not need */
	finishedFiles := []string{}
	for _, s := range rawFiles {
		if len(vitalData.Exceptlist) > 0 && vitalData.Exceptlist[s] {
			continue
		}
		finishedFiles = append(finishedFiles, s)
	}
	if !*localbin {
		for _, s := range finishedFiles {
			/* WHAT  A HACK -- ldd is really broken. HMM, did not used to be!*/
			if s == "" {
				continue
			}
			log_info("startExecution: not local walking '", s, "' full path is '", *root+s, "'")
			filepath.Walk(*root+s, walkFunc(pv, nil))
			log_info("finishedFiles is ", finishedFiles)
		}
	}
	/* build the library list given that we may have a different root */

	libList := strings.SplitN(*libs, ":", -1)
	rootedLibList := []string{}
	for _, s := range libList {
		log_info("startExecution: add lib ", s)
		rootedLibList = append(rootedLibList, fmt.Sprintf("%s/%s", *root, s))
	}
	/* this test could be earlier. We leave it all the way down here so we can 
	 * easily test the library code. Later, it can move
	 * earlier in the code. 
	 */
	if !vitalData.HostReady {
		log_info("Can not start jobs: ", vitalData.Error)
		return
	}
	log_info("startExecution: libList ", libList)
	ioProxyListenAddr := vitalData.HostAddr + ":" + ioProxyPort
	/* The ioProxy brings back the standard i/o streams from the slaves */
	workerChan, l, err := ioProxy(fam, ioProxyListenAddr, os.Stdout)
	if err != nil {
		log_error("startExecution: ioproxy: ", err)
	}

	req := StartReq{
		Command:         "e",
		Lfam:            l.Addr().Network(),
		Lserver:         l.Addr().String(),
		LocalBin:        *localbin,
		Args:            cmd,
		BytesToTransfer: pv.bytesToTransfer,
		LibList:         libList,
		Path:            *root,
		Nodes:           slaveNodes,
		Cmds:            pv.cmds,
		Cwd:             cwd,
	}

	r.Send("startExecution", req)
	resp := &Resp{}
	if r.Recv("startExecution", resp) != nil {
		log_error("Can't do start execution")
	}
	/* numWorkers tells us how many nodes will be connecting to our ioProxy */
	numWorkers := resp.NumNodes
	log_info("startExecution: waiting for ", numWorkers)
	for numWorkers > 0 {
		<-workerChan
		numWorkers--
		log_info("startExecution: read from a workerchan, numworkers = ", numWorkers)
	}
	log_info("startExecution: finished")
}

var (
	BadRangeErr = errors.New("bad range format")
)

type packVisitor struct {
	cmds            []*cmdToExec
	alreadyVisited  map[string]bool
	bytesToTransfer int64
}

func newPackVisitor() (p *packVisitor) {
	return &packVisitor{alreadyVisited: make(map[string]bool)}
}

func walkFunc(p *packVisitor, pv_err chan<- error) filepath.WalkFunc {
	return func(path string, info os.FileInfo, err error) error {
		if err != nil {
			if pv_err != nil {
				pv_err <- err
			}
			return nil
		}
		if info.IsDir() {
			if !p.VisitDir(path, info) {
				return filepath.SkipDir
			}
			return nil
		}
		p.VisitFile(path, info)
		return nil
	}
}

func (p *packVisitor) VisitDir(filePath string, f os.FileInfo) bool {
	filePath = strings.TrimSpace(filePath)
	filePath = strings.TrimRightFunc(filePath, isNull)

	if p.alreadyVisited[filePath] {
		return false
	}
	//	_, file := path.Split(filePath)
	uid := int(f.Sys().(*syscall.Stat_t).Uid)
	gid := int(f.Sys().(*syscall.Stat_t).Gid)
	var ftype int
	switch {
	case ((f.Mode() & os.ModeType) == 0):
		ftype = 0
	case f.IsDir():
		ftype = 1
	case ((f.Mode() & os.ModeSymlink) != 0):
		ftype = 2
	default:
		ftype = 3
	}
	perm := uint32(f.Mode().Perm())

	c := &cmdToExec{
		//		name: file,
		CurrentName: filePath,
		DestName:    filePath,
		Local:       0,
		Uid:         uid,
		Gid:         gid,
		Ftype:       ftype,
		Perm:        perm,
	}
	log_info("VisitDir: appending ", filePath, " ", []byte(filePath), " ", p.alreadyVisited)
	p.cmds = append(p.cmds, c)
	p.alreadyVisited[filePath] = true
	/* to make it possible to drag directories along, without dragging files along, we adopt that convention that 
	 * if the user ends a dir with /., then we won't recurse
	 */
	if strings.HasSuffix(filePath, "/.") {
		return false
	}
	return true
}

func isNull(r rune) bool {
	return r == 0
}

func (p *packVisitor) VisitFile(filePath string, f os.FileInfo) {
	// shouldn't need to do this, need to fix ldd
	filePath = strings.TrimSpace(filePath)
	filePath = strings.TrimRightFunc(filePath, isNull)
	if p.alreadyVisited[filePath] {
		return
	}

	uid := int(f.Sys().(*syscall.Stat_t).Uid)
	gid := int(f.Sys().(*syscall.Stat_t).Gid)
	var ftype int
	switch {
	case ((f.Mode() & os.ModeType) == 0):
		ftype = 0
	case f.IsDir():
		ftype = 1
	case ((f.Mode() & os.ModeSymlink) != 0):
		ftype = 2
	default:
		ftype = 3
	}
	perm := uint32(f.Mode().Perm())

	c := &cmdToExec{
		//		name: file,
		CurrentName: filePath,
		DestName:    filePath,
		Local:       0,
		Uid:         uid,
		Gid:         gid,
		Ftype:       ftype,
		Perm:        perm,
	}
	log_info("VisitFile: appending ", f.Name(), " ", f.Size(), " ", []byte(filePath), " ", p.alreadyVisited)

	p.cmds = append(p.cmds, c)

	switch {
	case f.Mode()&os.ModeType == 0:
		p.bytesToTransfer += f.Size()
	case f.Mode()&os.ModeSymlink != 0:
		/* we have to read the link but also get a path the file for 
		 * further walking. We think. 
		 */
		var walkPath string
		c.SymlinkTarget, walkPath = resolveLink(filePath)
		log_info("c.CurrentName", c.CurrentName, " filePath ", filePath)
		filepath.Walk(walkPath, walkFunc(p, nil))
	}
	p.alreadyVisited[filePath] = true
}

func resolveLink(filePath string) (linkPath, fullPath string) {
	// BUG: what about relative paths in the link?
	linkPath, err := os.Readlink(filePath)
	linkDir, linkFile := path.Split(linkPath)
	switch {
	case linkDir == "":
		linkDir, _ = path.Split(filePath)
	case linkDir[0] != '/':
		dir, _ := path.Split(filePath)
		linkDir = path.Join(dir, linkDir)
	}
	log_info("VisitFile: read link ", filePath, "->", linkDir+linkFile)
	if err != nil {
		log_error("VisitFile: readlink: ", err)
	}
	fullPath = path.Join(linkDir, linkFile)
	return
}
