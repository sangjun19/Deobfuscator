// Repository: dece-cash/go-dece
// File: cmd/gece/main.go

// Copyright 2014 The go-ethereum Authors
// This file is part of go-ethereum.
//
// go-ethereum is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// go-ethereum is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with go-ethereum. If not, see <http://www.gnu.org/licenses/>.

// gece is the official command-line client for Dece.
package main

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	godebug "runtime/debug"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/dece-cash/go-dece/czero/c_type"

	"github.com/dece-cash/go-dece/czero/deceparam"
	"github.com/dece-cash/go-dece/czero/superzk"

	"gopkg.in/urfave/cli.v1"

	"github.com/elastic/gosigar"
	"github.com/dece-cash/go-dece/accounts"
	"github.com/dece-cash/go-dece/accounts/keystore"
	"github.com/dece-cash/go-dece/cmd/utils"
	"github.com/dece-cash/go-dece/console"
	"github.com/dece-cash/go-dece/internal/debug"
	"github.com/dece-cash/go-dece/log"
	"github.com/dece-cash/go-dece/metrics"
	"github.com/dece-cash/go-dece/node"
	"github.com/dece-cash/go-dece/dece"
)

const (
	clientIdentifier = "gece" // Client identifier to advertise over the network
)

var (
	// Git SHA1 commit hash of the release (set via linker flags)
	gitCommit = ""
	// The app that holds all commands and flags.
	app = utils.NewApp(gitCommit, "the go-dece command line interface")
	// flags that configure the node
	nodeFlags = []cli.Flag{
		utils.IdentityFlag,
		utils.UnlockedAccountFlag,
		utils.PasswordFileFlag,
		utils.BootnodesFlag,
		utils.BootnodesV4Flag,
		utils.BootnodesV5Flag,
		utils.DataDirFlag,
		utils.KeyStoreDirFlag,
		utils.NoUSBFlag,
		utils.DashboardEnabledFlag,
		// utils.DashboardAddrFlag,
		// utils.DashboardPortFlag,
		// utils.DashboardRefreshFlag,
		utils.EthashCacheDirFlag,
		utils.EthashCachesInMemoryFlag,
		utils.EthashCachesOnDiskFlag,
		utils.EthashDatasetDirFlag,
		utils.EthashDatasetsInMemoryFlag,
		utils.EthashDatasetsOnDiskFlag,
		utils.TxPoolNoLocalsFlag,
		utils.TxPoolPriceLimitFlag,
		utils.TxPoolAccountSlotsFlag,
		utils.TxPoolGlobalSlotsFlag,
		utils.TxPoolAccountQueueFlag,
		utils.TxPoolGlobalQueueFlag,
		utils.TxPoolLifetimeFlag,
		utils.SyncModeFlag,
		utils.GCModeFlag,
		utils.CacheFlag,
		utils.CacheDatabaseFlag,
		utils.CacheGCFlag,
		utils.TrieCacheGenFlag,
		utils.ListenPortFlag,
		utils.MaxPeersFlag,
		utils.MaxPendingPeersFlag,
		utils.SerobaseFlag,
		utils.GasPriceFlag,
		utils.VThreadsFlag,
		utils.PThreadsFlag,
		utils.MinerThreadsFlag,
		utils.MiningEnabledFlag,
		utils.MineModeEnabledFlag,
		utils.TargetGasLimitFlag,
		utils.NATFlag,
		utils.NoDiscoverFlag,
		utils.DiscoveryV5Flag,
		utils.NetrestrictFlag,
		utils.NodeKeyFileFlag,
		utils.NodeKeyHexFlag,
		utils.DeveloperPeriodFlag,
		utils.DeveloperPasswordFlag,
		utils.AlphanetFlag,
		//utils.RinkebyFlag,
		utils.ExchangeFlag,
		utils.ExchangeValueStrFlag,
		utils.StakeFlag,
		utils.AutoMergeFlag,
		utils.ConfirmedBlockFlag,
		utils.RecordBlockShareNumber,
		utils.LightNodeFlag,
		utils.ResetBlockNumber,

		utils.DeveloperFlag,
		utils.OfflineFlag,
		utils.SnapshotFlag,
		utils.TestStartBlockFlag,
		utils.TestForkFlag,
		utils.VMEnableDebugFlag,
		utils.NetworkIdFlag,
		utils.RPCCORSDomainFlag,
		utils.RPCVirtualHostsFlag,
		utils.SeroStatsURLFlag,
		utils.MetricsEnabledFlag,
		utils.FakePoWFlag,
		utils.NoCompactionFlag,
		utils.GpoBlocksFlag,
		utils.GpoPercentileFlag,
		utils.ExtraDataFlag,
		configFileFlag,
	}

	proofFlags = []cli.Flag{
		utils.ProofEnabledFlag,
		utils.ProofMaxThreadFlag,
		utils.ProofMaxQueueFlag,
		utils.ProofzinFeeFlag,
		utils.ProofoinFeeFlag,
		utils.ProofoutFeeFlag,
		utils.ProofFixedFeeFlag,
	}

	rpcFlags = []cli.Flag{
		utils.RPCEnabledFlag,
		utils.RPCListenAddrFlag,
		utils.RPCPortFlag,
		utils.RPCApiFlag,
		utils.RPCRequestContentLength,
		utils.RPCReadTimeoutFlag,
		utils.RPCWriteTimeoutFlag,
		utils.RPCIdleTimeoutFlag,
		utils.WSEnabledFlag,
		utils.WSListenAddrFlag,
		utils.WSPortFlag,
		utils.WSApiFlag,
		utils.WSAllowedOriginsFlag,
		utils.IPCDisabledFlag,
		utils.IPCPathFlag,
	}

	metricsFlags = []cli.Flag{
		utils.MetricsEnableInfluxDBFlag,
		utils.MetricsInfluxDBEndpointFlag,
		utils.MetricsInfluxDBDatabaseFlag,
		utils.MetricsInfluxDBUsernameFlag,
		utils.MetricsInfluxDBPasswordFlag,
		utils.MetricsInfluxDBHostTagFlag,
	}
)

func init() {
	// Initialize the CLI app and start Dece
	app.Action = gece
	app.HideVersion = true // we have a command to print the version
	app.Copyright = "Copyright 2013-2018 The go-dece Authors"
	app.Commands = []cli.Command{
		// See chaincmd.go:
		importCommand,
		exportCommand,
		importPreimagesCommand,
		exportPreimagesCommand,
		copydbCommand,
		removedbCommand,
		//dumpCommand,
		// See monitorcmd.go:
		monitorCommand,
		// See accountcmd.go:
		accountCommand,
		// See consolecmd.go:
		consoleCommand,
		attachCommand,
		javascriptCommand,
		// See misccmd.go:
		makecacheCommand,
		makedagCommand,
		versionCommand,
		bugCommand,
		licenseCommand,
		// See config.go
		dumpConfigCommand,
	}
	sort.Sort(cli.CommandsByName(app.Commands))

	app.Flags = append(app.Flags, nodeFlags...)
	app.Flags = append(app.Flags, rpcFlags...)
	app.Flags = append(app.Flags, consoleFlags...)
	app.Flags = append(app.Flags, debug.Flags...)
	app.Flags = append(app.Flags, proofFlags...)
	app.Flags = append(app.Flags, metricsFlags...)

	app.Before = func(ctx *cli.Context) error {

		subCommandName := ""

		if len(ctx.Args()) > 0 {
			subCommandName = ctx.Args()[0]
		}

		if !strings.EqualFold(subCommandName, "attach") && !strings.EqualFold(subCommandName, "version") {
			netType := c_type.NET_Beta
			switch {
			case ctx.GlobalBool(utils.AlphanetFlag.Name):
				netType = c_type.NET_Alpha
			case ctx.GlobalBool(utils.DeveloperFlag.Name):
				netType = c_type.NET_Dev
			}
			superzk.ZeroInit(getKeyStore(ctx), netType)

		}

		runtime.GOMAXPROCS(runtime.NumCPU())

		logdir := ""
		if ctx.GlobalBool(utils.DashboardEnabledFlag.Name) {
			logdir = (&node.Config{DataDir: utils.MakeDataDir(ctx)}).ResolvePath("logs")
		}
		if err := debug.Setup(ctx, logdir); err != nil {
			return err
		}
		// Cap the cache allowance and tune the garbage collector
		var mem gosigar.Mem
		if err := mem.Get(); err == nil {
			allowance := int(mem.Total / 1024 / 1024 / 3)
			if cache := ctx.GlobalInt(utils.CacheFlag.Name); cache > allowance {
				log.Warn("Sanitizing cache to Go's GC limits", "provided", cache, "updated", allowance)
				ctx.GlobalSet(utils.CacheFlag.Name, strconv.Itoa(allowance))
			}
		}
		// Ensure Go's GC ignores the database cache for trigger percentage
		cache := ctx.GlobalInt(utils.CacheFlag.Name)
		gogc := math.Max(20, math.Min(100, 100/(float64(cache)/1024)))

		log.Debug("Sanitizing Go's GC trigger", "percent", int(gogc))
		godebug.SetGCPercent(int(gogc))

		// Start metrics export if enabled
		utils.SetupMetrics(ctx)

		// Start system runtime metrics collection
		go metrics.CollectProcessMetrics(3 * time.Second)

		utils.SetupNetwork(ctx)
		return nil
	}

	app.After = func(ctx *cli.Context) error {
		debug.Exit()
		console.Stdin.Close() // Resets terminal mode.
		return nil
	}
}

func getKeyStore(ctx *cli.Context) string {
	dataDir := node.DefaultDataDir()
	switch {
	case ctx.GlobalIsSet(utils.DataDirFlag.Name):
		dataDir = ctx.GlobalString(utils.DataDirFlag.Name)
	case ctx.GlobalBool(utils.AlphanetFlag.Name):
		dataDir = filepath.Join(node.DefaultDataDir(), "alpha")
	case ctx.GlobalBool(utils.DeveloperFlag.Name):
		dataDir = filepath.Join(node.DefaultDataDir(), "dev")
	}
	keyStoreDir := ""

	if ctx.GlobalIsSet(utils.KeyStoreDirFlag.Name) {
		keyStoreDir = ctx.GlobalString(utils.KeyStoreDirFlag.Name)
	}
	var (
		keydir string
	)
	switch {
	case filepath.IsAbs(keyStoreDir):
		keydir = keyStoreDir
	case dataDir != "":
		if keyStoreDir == "" {
			keydir = filepath.Join(dataDir, "keystore")
		} else {
			keydir, _ = filepath.Abs(keyStoreDir)
		}
	case keyStoreDir != "":
		keydir, _ = filepath.Abs(keyStoreDir)
	}

	return keydir
}


func main() {
	if err := app.Run(os.Args); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// gece is the main entry point into the system if no special subcommand is ran.
// It creates a default node based on the command line arguments and runs it in
// blocking mode, waiting for it to be shut down.
func gece(ctx *cli.Context) error {
	if args := ctx.Args(); len(args) > 0 {
		return fmt.Errorf("invalid command: %q", args[0])
	}
	node := makeFullNode(ctx)
	startNode(ctx, node)
	node.Wait()
	return nil
}

// startNode boots up the system node and all registered protocols, after which
// it unlocks any requested accounts, and starts the RPC/IPC interfaces and the
// miner.
func startNode(ctx *cli.Context, stack *node.Node) {

	debug.Memsize.Add("node", stack)

	// Start up the node itself
	utils.StartNode(stack)

	// Unlock any account specifically requested
	ks := stack.AccountManager().Backends(keystore.KeyStoreType)[0].(*keystore.KeyStore)

	if deceparam.Is_Dev() && ctx.GlobalString(utils.DeveloperPasswordFlag.Name) != "" {
		for _, wallet := range ks.Wallets() {
			err := ks.Unlock(wallet.Accounts()[0], ctx.GlobalString(utils.DeveloperPasswordFlag.Name))
			if err != nil {
				fmt.Printf("unclock %v failed,%v", wallet.Accounts()[0].Address.String(), err)
			}
		}
	} else {
		passwords := utils.MakePasswordList(ctx)
		unlocks := strings.Split(ctx.GlobalString(utils.UnlockedAccountFlag.Name), ",")
		for i, account := range unlocks {
			if trimmed := strings.TrimSpace(account); trimmed != "" {
				unlockAccount(ctx, ks, trimmed, i, passwords)
			}
		}
	}

	// Register wallet event handlers to open and auto-derive wallets
	events := make(chan accounts.WalletEvent, 16)
	stack.AccountManager().Subscribe(events)

	go func() {
		// Create a chain state reader for self-derivation
		//rpcClient, err := stack.Attach()
		//if err != nil {
		//	utils.Fatalf("Failed to attach to self: %v", err)
		//}
		//stateReader := dececlient.NewClient(rpcClient)

		// Open any wallets already attached
		for _, wallet := range stack.AccountManager().Wallets() {
			if err := wallet.Open(""); err != nil {
				log.Warn("Failed to open wallet", "url", wallet.URL(), "err", err)
			}
		}
		// Listen for wallet event till termination
		for event := range events {
			switch event.Kind {
			case accounts.WalletArrived:
				if err := event.Wallet.Open(""); err != nil {
					log.Warn("New wallet appeared, failed to open", "url", event.Wallet.URL(), "err", err)
				}

			case accounts.WalletDropped:
				log.Info("Old wallet dropped", "url", event.Wallet.URL())
				event.Wallet.Close()
			}
		}
	}()
	// Start auxiliary services if enabled
	if ctx.GlobalBool(utils.MiningEnabledFlag.Name) {
		// Mining only makes sense if a full Dece node is running
		if ctx.GlobalString(utils.SyncModeFlag.Name) == "light" {
			utils.Fatalf("Light clients do not support mining")
		}
		var dece *dece.Dece
		if err := stack.Service(&dece); err != nil {
			utils.Fatalf("Dece service not running: %v", err)
		}
		// Use a reduced number of threads if requested
		if threads := ctx.GlobalInt(utils.MinerThreadsFlag.Name); threads > 0 {
			type threaded interface {
				SetThreads(threads int)
			}
			if th, ok := dece.Engine().(threaded); ok {
				th.SetThreads(threads)
			}
		}
		// Set the gas price to the limits from the CLI and start mining
		dece.TxPool().SetGasPrice(utils.GlobalBig(ctx, utils.GasPriceFlag.Name))
		if err := dece.StartMining(true); err != nil {
			utils.Fatalf("Failed to start mining: %v", err)
		}
	}
}
