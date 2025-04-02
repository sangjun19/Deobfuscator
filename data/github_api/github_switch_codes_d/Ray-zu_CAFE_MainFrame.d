// Repository: Ray-zu/CAFE
// File: source/gui/controls/MainFrame.d

/+ ------------------------------------------------------------ +
 + Author : aoitofu <aoitofu@dr.com>                            +
 + This is part of CAFE ( https://github.com/aoitofu/CAFE ).    +
 + ------------------------------------------------------------ +
 + Please see /LICENSE.                                         +
 + ------------------------------------------------------------ +/
module cafe.gui.controls.MainFrame;
import cafe.app,
       cafe.config,
       cafe.gui.Action,
       cafe.gui.controls.AppInfoPanel,
       cafe.gui.controls.BMPViewer,
       cafe.gui.controls.CafeConfDialog,
       cafe.gui.controls.ConfigDialogs,
       cafe.gui.controls.ConfigTabs,
       cafe.gui.controls.ErrorPanel,
       cafe.gui.controls.PreviewPlayer,
       cafe.gui.controls.FragmentsExplorer,
       cafe.gui.controls.FileDialogs,
       cafe.gui.controls.StartPanel,
       cafe.gui.controls.TimelineTabs,
       cafe.project.Project;
import std.conv,
       std.file,
       std.format,
       std.json;
import dlangui,
       dlangui.dialogs.dialog;

class MainFrame : AppFrame
{
    enum Layout = q{
        HorizontalLayout {
            VerticalLayout {
                HorizontalLayout {
                    PreviewPlayer { id:preview }
                    FragmentsExplorer { id:flagexp }
                }
                TimelineTabs { id:timeline }
            }
            ConfigTabs { id:tabs }
        }
    };

    static @property PreviewHeight ()
    {
        return config( "layout/main/PreviewHeight" ).uintegerDef( 350 ).to!int;
    }
    static @property ConfigWidth ()
    {
        return config( "layout/main/ConfigWidth" ).uintegerDef( 400 ).to!int;
    }

    private:
        bool initialized = false;

        MenuItem top_menu;

        PreviewPlayer     preview;
        TimelineTabs      timeline;
        ConfigTabs        tabs;
        FragmentsExplorer fragexp;

        string last_saved_file;

        auto open ()
        {
            auto dlg = new FileOpenDialog( UIString.fromRaw("Open project"), window );
            dlg.dialogResult = delegate ( Dialog d, const Action a )
            {
                if ( a.id != ACTION_OPEN.id ) return;
                auto file = dlg.filename;
                auto text = file.readText;
                Cafe.instance.curProject = new Project( parseJSON(text) );
                last_saved_file = file;
            };
            dlg.show;
        }

        auto save ()
        {
            if ( !Cafe.instance.curProject ) return;
            if ( last_saved_file.exists ) {
                auto text = Cafe.instance.curProject.json.to!string;
                last_saved_file.write( text );
            } else saveAs;
        }

        auto saveAs ()
        {
            if ( !Cafe.instance.curProject ) return;
            auto dlg = new FileSaveDialog( UIString.fromRaw("Save project"), window );
            dlg.dialogResult = delegate ( Dialog d, const Action a )
            {
                if ( a.id != ACTION_SAVE.id ) return;
                auto file = dlg.filename;
                auto text = Cafe.instance.curProject.json.to!string;
                file.write( text );
                last_saved_file = file;
            };
            dlg.show;
        }

        /+ プロジェクトのインスタンスが変更された時 +/
        auto projectRefresh ()
        {
            auto p = Cafe.instance.curProject;
            preview.project = p;
            timeline.project = p;
            tabs.propertyEditor.project = p;
            tabs.componentTree .project = p;

            if ( p ) {
            } else {
                new StartPanel( window ).show;
            }
            handleAction( Action_PreviewRefresh );
            handleAction( Action_ObjectRefresh  );
            return true;
        }

    protected:
        override void initialize ()
        {
            super.initialize();
            _appName = AppName;
            CafeConf.load( settingsDir~"/config.json" );
        }

        override Widget createBody ()
        {
            auto w = parseML( Layout );
            preview  = cast(PreviewPlayer)    w.childById( "preview" );
            timeline = cast(TimelineTabs)     w.childById( "timeline" );
            tabs     = cast(ConfigTabs)       w.childById( "tabs" );
            fragexp  = cast(FragmentsExplorer)w.childById( "flagexp" );
            return w;
        }

        override MainMenu createMainMenu ()
        {
            top_menu = new MenuItem;

            auto menu = new MenuItem( new Action( 1, "TopMenu_File" ) );
            with ( menu ) {
                add( Action_ProjectNew    );
                add( Action_ProjectOpen   );
                add( Action_ProjectSave   );
                add( Action_ProjectSaveAs );
                add( Action_ProjectClose  );
            }
            top_menu.add( menu );

            with ( menu = new MenuItem( new Action( 1, "TopMenu_Play" ) ) ) {
                add( Action_PreviewRefresh );

                add( Action_Play  );
                add( Action_Pause );
                add( Action_Stop  );

                add( Action_MoveBehind );
                add( Action_MoveAHead  );

                add( Action_ShiftBehind );
                add( Action_ShiftAHead  );
            }
            top_menu.add( menu );

            with ( menu = new MenuItem( new Action( 1, "TopMenu_Info" ) ) ) {
                add( Action_Configure  );
                add( Action_VersionDlg );
                add( Action_HomePage   );
            }
            top_menu.add( menu );

            return new MainMenu( top_menu );
        }

        override ToolBarHost createToolbars ()
        {
            auto host = new ToolBarHost;
            auto bar  = host.getOrAddToolbar( "File" );
            with ( bar ) {
                addButtons( Action_ProjectSave   );
                addButtons( Action_ProjectSaveAs );
            }
            with ( bar = host.getOrAddToolbar( "Play" ) ) {
                addButtons( Action_Play  );
                addButtons( Action_Pause );
                addButtons( Action_Stop  );

                addButtons( Action_MoveBehind );
                addButtons( Action_MoveAHead  );

                addButtons( Action_ShiftBehind );
                addButtons( Action_ShiftAHead  );
            }
            return host;
        }

    public:
        this ()
        {
            super();
            handleAction( new Action_UpdateStatus( "Status_Boot" ) );
            last_saved_file = "";
        }

        /+ ウィンドウ破棄リクエストを受理するかどうか +/
        auto canClose ()
        {
            CafeConf.save;
            return true;
        }

        override void measure ( int w, int h )
        {
            preview.minHeight  = PreviewHeight;
            timeline.maxHeight = h - preview.minHeight;
            tabs.minWidth      = ConfigWidth;
            super.measure( w, h );
        }

        override void onDraw ( DrawBuf b )
        {
            /+ 最初の描画の時に行う +/
            if ( !initialized )
                projectRefresh;
            initialized = true;
            super.onDraw( b );
        }

        override bool handleAction ( const Action a )
        {
            if ( !a ) return false;

            try {
            import cafe.gui.Action;
            switch ( a.id ) with( EditorActions ) {
                case UpdateStatus:
                    statusLine.setStatusText( a.label );
                    return true;

                case ProjectNew:
                    new ProjectConfigDialog( true, window ).show;
                    return true;
                case ProjectOpen:
                    open;
                    return true;
                case ProjectSave:
                    save;
                    return true;
                case ProjectSaveAs:
                    saveAs;
                    return true;
                case ProjectClose:
                    Cafe.instance.curProject = null;
                    return true;

                case ProjectRefresh:
                    return projectRefresh;
                case PreviewRefresh:
                    return preview.handleAction( a );
                case ObjectRefresh:
                    tabs.propertyEditor.updateWidgets;
                    return true;
                case CompTreeRefresh:
                    return tabs.componentTree.handleAction( a );
                case TimelineRefresh:
                    timeline.updateWidgets;
                    return true;

                case ChangeFrame:
                    handleAction( Action_ObjectRefresh );
                    handleAction( Action_PreviewRefresh );
                    return true;

                case CompTreeOpen:
                    timeline.addTab(
                            tabs.componentTree.items.selectedItem.id );
                    return true;

                case Configure:
                    new CafeConfDialog( window ).show;
                    return true;
                case VersionDlg:
                    new AppInfoPanel( window ).show;
                    return true;
                case HomePage:
                    Platform.instance.openURL( AppURL );
                    return true;

                default:
                    return super.handleAction( a );
            }
            } catch ( Throwable e ) {
                new ErrorPanel( e, window ).show;
                return false;
            }
        }
}
