#include "cbase.h"

#include "view_scene.h"
#include "text_message.h"
#include <engine/IEngineSound.h>
#include "c_user_message_register.h"
#include "cdll_client_int.h"
#include "IGameUIFuncs.h"

#include "npcs/zmr_zombiebase_shared.h"
#include "ui/zmr_hud_chat.h"
#include "ui/zmr_hud_tooltips.h"

#include "c_zmr_tips.h"
#include "c_zmr_util.h"

// memdbgon must be the last include file in a .cpp file!!!
#include "tier0/memdbgon.h"


ConVar zm_hudchat_color( "zm_hudchat_color", "c13c3c", FCVAR_ARCHIVE );
ConVar zm_hudchat_color_zm( "zm_hudchat_color_zm", "49ff59", FCVAR_ARCHIVE );



int UTIL_CreateClientModel( const char* pszModel )
{
    int index;
    index = modelinfo->GetModelIndex( pszModel );

    if ( index == -1 )
    {
        // No model found, register our own.
        index = modelinfo->RegisterDynamicModel( pszModel, true );
    }
    
    return index;
}

bool UTIL_PassKeyToEngine( ButtonCode_t code, bool bDoRelease = true, bool bOnlyRelease = false )
{
    Assert( bDoRelease || !bOnlyRelease );


    const char* binding = gameuifuncs->GetBindingForButtonCode( code );
    if ( binding && *binding )
    {
        if ( !bOnlyRelease )
        {
            engine->ClientCmd_Unrestricted( binding );
        }

        // HACK: Call '-' version of the command to unpress said key.
        if ( bDoRelease && binding[0] == '+' )
        {
            char temp[128];
            Q_strncpy( temp, binding, sizeof( temp ) );
            temp[0] = '-';

            engine->ClientCmd_Unrestricted( temp );
        }

        return true;
    }

    return false;
}

void UTIL_ParseColorFromString( const char* str, int clr[], int nColors )
{
    Assert( str );

    CSplitString split( str, " " );

    Q_memset( clr, 0, nColors * sizeof( int ) );

    for ( int i = 0; i < nColors; i++ )
    {
        if ( split.Count() > i )
            clr[i] = Q_atoi( split[i] );
        else
            break;
    }
}

void UTIL_ParseFloatColorFromString( const char* str, float clr[], int nColors )
{
    Assert( str );

    CSplitString split( str, " " );

    Q_memset( clr, 0, nColors * sizeof( float ) );

    for ( int i = 0; i < nColors; i++ )
    {
        if ( split.Count() > i )
            clr[i] = Q_atof( split[i] );
        else
            break;
    }
}



void ZMClientUtil::PrintNotify( const char* msg, ZMChatNotifyType_t type )
{
    char buf[7];
    GetNotifyTypeColor( type, buf, ARRAYSIZE( buf ) );

    if ( msg[0] == '#' )
    {
        ZMClientUtil::ChatPrint( GetLocalPlayerIndex(), true, "\x07%s%s", buf, g_pVGuiLocalize->FindAsUTF8( msg ) );
    }
    else
    {
        ZMClientUtil::ChatPrint( GetLocalPlayerIndex(), true, "\x07%s%s", buf, msg );
    }
    
}

void ZMClientUtil::GetNotifyTypeColor( ZMChatNotifyType_t type, char* buffer, size_t len )
{
    const char* pColor = nullptr;

    switch ( type )
    {
    case ZMCHATNOTIFY_ZM : pColor = zm_hudchat_color_zm.GetString(); break;
    default : pColor = zm_hudchat_color.GetString(); break;
    }

    if ( pColor )
        Q_strncpy( buffer, pColor, len );
}

void ZMClientUtil::ChatPrint( int iPlayerIndex, bool bPlaySound, const char* format, ... )
{
    CHudChat* pChat = GET_HUDELEMENT( CHudChat );

    if ( !pChat ) return;



    if ( format[0] == '#' )
    {
        pChat->ChatPrintf( iPlayerIndex, CHAT_FILTER_NONE, "%s",
            g_pVGuiLocalize->FindAsUTF8( format ) );
    }
    else
    {
        va_list marker;
        char msg[512];

        va_start( marker, format );
        Q_vsnprintf( msg, sizeof( msg ), format, marker );
        va_end( marker );

        pChat->ChatPrintf( iPlayerIndex, CHAT_FILTER_NONE, "%s", msg );
    }

    if ( bPlaySound )
    {
        CLocalPlayerFilter filter;
        C_BaseEntity::EmitSound( filter, SOUND_FROM_LOCAL_PLAYER, "HudChat.Message" );
    }
}

void ZMClientUtil::QueueTooltip( const char* name, float delay )
{
    CZMHudTooltip* tips = GET_HUDELEMENT( CZMHudTooltip );

    if ( tips )
    {
        tips->QueueTip( name, delay );
    }
}

int ZMClientUtil::ShowTooltipByName( const char* szName, bool force )
{
    CZMHudTooltip* tips = GET_HUDELEMENT( CZMHudTooltip );

    if ( tips )
    {
        return tips->SetMessageByName( szName, force );
    }

    return 0;
}

void ZMClientUtil::ShowTooltip( const char* szMsg )
{
    CZMHudTooltip* tips = GET_HUDELEMENT( CZMHudTooltip );

    if ( tips )
    {
        tips->SetMessage( szMsg );
    }
}

void ZMClientUtil::HideTooltip( int index )
{
    CZMHudTooltip* tips = GET_HUDELEMENT( CZMHudTooltip );

    if ( tips )
    {
        if ( tips->GetCurrentIndex() == index )
        {
            tips->HideTooltip();
        }
    }
}

void ZMClientUtil::HideTooltip()
{
    CZMHudTooltip* tips = GET_HUDELEMENT( CZMHudTooltip );

    if ( tips )
    {
        tips->HideTooltip();
    }
}

bool ZMClientUtil::WorldToScreen( const Vector& pos, Vector& screen, int& x, int& y )
{
    int behind = ScreenTransform( pos, screen );

    if ( !behind )
    {
        int w = ScreenWidth();
        int h = ScreenHeight();
        x =  0.5f * screen[0] * w;
        y = -0.5f * screen[1] * h;
        x += 0.5f * w;
        y += 0.5f * h;
    }

    return !behind ? true : false;
}

int ZMClientUtil::GetSelectedZombieCount()
{
    int myindex = GetLocalPlayerIndex();

    if ( !myindex ) return 0;


    int num = 0;

    g_ZombieManager.ForEachSelectedZombie( myindex, [ &num ]( C_ZMBaseZombie* pZombie )
    {
        ++num;
    } );

    return num;
}

void ZMClientUtil::SelectAllZombies( bool bSendCommand )
{
    int index = GetLocalPlayerIndex();
    if ( !index ) return;



    g_ZombieManager.ForEachZombie( [ index ]( C_ZMBaseZombie* pZombie )
    {
        pZombie->SetSelector( index );
    } );


    if ( bSendCommand )
    {
        // Tell server we're selecting all.
        engine->ClientCmd( "zm_cmd_selectall" );
    }
}

void ZMClientUtil::DeselectAllZombies( bool bSendCommand )
{
    int index = GetLocalPlayerIndex();
    if ( !index ) return;



    g_ZombieManager.ForEachSelectedZombie( index, []( C_ZMBaseZombie* pZombie )
    {
        pZombie->SetSelector( 0 );
    } );


    if ( bSendCommand )
    {
        // Tell server we're deselecting.
        engine->ClientCmd( "zm_cmd_unselectall" );
    }
}

void ZMClientUtil::SelectSingleZombie( C_ZMBaseZombie* pZombie, bool bSticky )
{
    if ( !bSticky )
    {
        DeselectAllZombies( false );
    }

    int index = GetLocalPlayerIndex();

    engine->ClientCmd( VarArgs( "zm_cmd_select %i%s",
        pZombie->entindex(),
        bSticky ? " 1" : "" ) );

    pZombie->SetSelector( index );
}

void ZMClientUtil::SelectZombiesInRadius( const Vector& vecPos, float radius, bool bSticky )
{
    float flRadiusSqr = radius * radius;
    CUtlVector<CZMBaseZombie*> vZombies;


    g_ZombieManager.ForEachZombie( [ &vecPos, flRadiusSqr, &vZombies ]( C_ZMBaseZombie* pZombie )
    {
        if ( pZombie->GetAbsOrigin().DistToSqr( vecPos ) < flRadiusSqr )
        {
            vZombies.AddToTail( pZombie );
        }
    } );

    SelectZombies( vZombies, bSticky );
}

void ZMClientUtil::SelectZombies( const CUtlVector<C_ZMBaseZombie*>& vZombies, bool bSticky )
{
    C_ZMBaseZombie* pZombie;
    int index = GetLocalPlayerIndex();


    if ( !index ) return;

    if ( !bSticky )
    {
        // Send command if we didn't select anything.
        DeselectAllZombies( !vZombies.Count() );
    }


    // We didn't select anything.
    if ( !vZombies.Count() )
    {
        return;
    }


    // We need to send selection commands in batches.
    // It's very easy to go over the argument / command size limit.
    const int nBatchLimit = 30;

    const int nTotalCount = vZombies.Count();


    char cmdbuffer[512];
    int i = 0;
    
    int count = MIN( nBatchLimit, nTotalCount );

    do
    {
        cmdbuffer[0] = '\0';


        for ( ; i < count; i++ )
        {
            pZombie = vZombies.Element( i );

            pZombie->SetSelector( index );
            
            char add[16]; // "XXXXX "
            int len = Q_snprintf( add, sizeof( add ), "%i ", pZombie->entindex() );

            Q_strncat( cmdbuffer, add, sizeof( cmdbuffer ), len );
        }

        if ( cmdbuffer[0] )
        {
            const char* cmd = VarArgs( "zm_cmd_selectmult %s %s",  bSticky ? "1": "0", cmdbuffer );
            //DevMsg( "Sending select multiple command: \"%s\"\n", cmd );
            engine->ClientCmd( cmd );
        }


        count += nBatchLimit;
        if ( count > nTotalCount )
            count = nTotalCount;

        // Send it as a sticky in the next batch.
        bSticky = true;
    }
    while ( i < nTotalCount );
}

int ZMClientUtil::GetGroupZombieCount( int group )
{
    int num = 0;

    g_ZombieManager.ForEachZombie( [ group, &num ]( C_ZMBaseZombie* pZombie )
    {
        if ( pZombie->GetGroup() == group )
        {
            ++num;
        }
    } );

    return num;
}

void ZMClientUtil::SetSelectedGroup( int group )
{
    int index = GetLocalPlayerIndex();

    if ( !index ) return;


    g_ZombieManager.ForEachZombie( [ group, index ]( C_ZMBaseZombie* pZombie )
    {
        if ( pZombie->GetSelectorIndex() == index )
        {
            pZombie->SetGroup( group );
        }
        else if ( pZombie->GetGroup() == group )
        {
            pZombie->SetGroup( INVALID_GROUP_INDEX );
        }
    } );
}

void ZMClientUtil::SelectGroup( int group, bool force )
{
    int index = GetLocalPlayerIndex();

    if ( !index ) return;


    CUtlVector<C_ZMBaseZombie*> vSelect;
    vSelect.Purge();


    if ( !force && GetGroupZombieCount( group ) == 0 )
        return;

    // Don't send selection if we've already selected that group.
    bool bNew = false;
    
    g_ZombieManager.ForEachZombie( [ &bNew, &vSelect, group, index ]( C_ZMBaseZombie* pZombie )
    {
        if ( pZombie->GetGroup() == group )
            vSelect.AddToTail( pZombie );

        if ((pZombie->GetGroup() == group && pZombie->GetSelectorIndex() != index)
        ||  (pZombie->GetGroup() != group && pZombie->GetSelectorIndex() == index))
        {
            bNew = true;
        }
    } );

    if ( bNew )
    {
        DeselectAllZombies( false );
        SelectZombies( vSelect, false );
    }
}

void __MsgFunc_ZMChatNotify( bf_read& msg )
{
    ZMChatNotifyType_t type = (ZMChatNotifyType_t)msg.ReadByte();

    char buf[512];
    msg.ReadString( buf, sizeof( buf ) );

    ZMClientUtil::PrintNotify( buf, type );
}

USER_MESSAGE_REGISTER( ZMChatNotify );
