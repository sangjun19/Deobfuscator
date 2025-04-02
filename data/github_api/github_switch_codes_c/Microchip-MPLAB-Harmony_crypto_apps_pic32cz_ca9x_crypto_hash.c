/*******************************************************************************
  MPLAB Harmony Application Source File

  Company:
    Microchip Technology Inc.

  File Name:
    crypto_hash.c

  Summary:
    This file contains the source code for the MPLAB Harmony application.

  Description:
    This file contains the source code for the MPLAB Harmony application.  It
    implements the logic of the application's state machine and it may call
    API routines of other MPLAB Harmony modules in the system, such as drivers,
    system services, and middleware.  However, it does not call any of the
    system interfaces (such as the "Initialize" and "Tasks" functions) of any of
    the modules in the system or make any assumptions about when those functions
    are called.  That is the responsibility of the configuration-specific system
    files.
*******************************************************************************/

/*******************************************************************************
* Copyright (C) 2025 Microchip Technology Inc. and its subsidiaries.
*
* Subject to your compliance with these terms, you may use Microchip software
* and any derivatives exclusively with Microchip products. It is your
* responsibility to comply with third party license terms applicable to your
* use of third party software (including open source software) that may
* accompany Microchip software.
*
* THIS SOFTWARE IS SUPPLIED BY MICROCHIP "AS IS". NO WARRANTIES, WHETHER
* EXPRESS, IMPLIED OR STATUTORY, APPLY TO THIS SOFTWARE, INCLUDING ANY IMPLIED
* WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, AND FITNESS FOR A
* PARTICULAR PURPOSE.
*
* IN NO EVENT WILL MICROCHIP BE LIABLE FOR ANY INDIRECT, SPECIAL, PUNITIVE,
* INCIDENTAL OR CONSEQUENTIAL LOSS, DAMAGE, COST OR EXPENSE OF ANY KIND
* WHATSOEVER RELATED TO THE SOFTWARE, HOWEVER CAUSED, EVEN IF MICROCHIP HAS
* BEEN ADVISED OF THE POSSIBILITY OR THE DAMAGES ARE FORESEEABLE. TO THE
* FULLEST EXTENT ALLOWED BY LAW, MICROCHIP'S TOTAL LIABILITY ON ALL CLAIMS IN
* ANY WAY RELATED TO THIS SOFTWARE WILL NOT EXCEED THE AMOUNT OF FEES, IF ANY,
* THAT YOU HAVE PAID DIRECTLY TO MICROCHIP FOR THIS SOFTWARE.
*******************************************************************************/
 
// *****************************************************************************
// *****************************************************************************
// Section: Included Files
// *****************************************************************************
// *****************************************************************************

#include "crypto/common_crypto/crypto_common.h"
#include "crypto/common_crypto/crypto_hash.h"
#include "crypto/wolfcrypt/crypto_hash_wc_wrapper.h"

// *****************************************************************************
// *****************************************************************************
// Section: Global Data Definitions
// *****************************************************************************
// *****************************************************************************

#define CRYPTO_HASH_SESSION_MAX (1) 

// *****************************************************************************
// *****************************************************************************
// Section: Function Definitions
// *****************************************************************************
// *****************************************************************************

//MD5 Algorithm
crypto_Hash_Status_E Crypto_Hash_Md5_Digest(crypto_HandlerType_E md5Handler_en, uint8_t *ptr_data, uint32_t dataLen, uint8_t *ptr_digest, uint32_t md5SessionId)
{
   	crypto_Hash_Status_E ret_md5Stat_en = CRYPTO_HASH_ERROR_NOTSUPPTED;

    if( (ptr_data == NULL) || (dataLen == 0u) )
    {
        ret_md5Stat_en = CRYPTO_HASH_ERROR_INPUTDATA;
    }
    else if(ptr_digest == NULL)
    {      
        ret_md5Stat_en = CRYPTO_HASH_ERROR_OUTPUTDATA;
    }
    else if( (md5SessionId <= 0u) || (md5SessionId > (uint32_t)CRYPTO_HASH_SESSION_MAX) )
    {
        ret_md5Stat_en = CRYPTO_HASH_ERROR_SID;
    }
    else
    {
        switch(md5Handler_en)
        {
            case CRYPTO_HANDLER_SW_WOLFCRYPT:
                ret_md5Stat_en = Crypto_Hash_Wc_Md5Digest(ptr_data, dataLen, ptr_digest);
                break;
            default:
                ret_md5Stat_en = CRYPTO_HASH_ERROR_HDLR;
                break;
        };
    }
	return ret_md5Stat_en;  
}

crypto_Hash_Status_E Crypto_Hash_Md5_Init(st_Crypto_Hash_Md5_Ctx *ptr_md5Ctx_st, crypto_HandlerType_E md5HandlerType_en, uint32_t md5SessionId)
{
	crypto_Hash_Status_E ret_md5Stat_en = CRYPTO_HASH_ERROR_NOTSUPPTED;

    if(ptr_md5Ctx_st == NULL)
    {
        ret_md5Stat_en = CRYPTO_HASH_ERROR_CTX;
    }
    else if( (md5SessionId <= 0u) || (md5SessionId > (uint32_t)CRYPTO_HASH_SESSION_MAX) )
    {
        ret_md5Stat_en = CRYPTO_HASH_ERROR_SID;
    }
    else
    {
        ptr_md5Ctx_st->md5SessionId = md5SessionId;
        ptr_md5Ctx_st->md5Handler_en = md5HandlerType_en;
        
        switch(ptr_md5Ctx_st->md5Handler_en)
        {
            case CRYPTO_HANDLER_SW_WOLFCRYPT:
                ret_md5Stat_en = Crypto_Hash_Wc_Md5Init((void*)ptr_md5Ctx_st->arr_md5DataCtx);
                break;
            default:
                ret_md5Stat_en = CRYPTO_HASH_ERROR_HDLR;
                break;
        };
    }
	return ret_md5Stat_en;
}

crypto_Hash_Status_E Crypto_Hash_Md5_Update(st_Crypto_Hash_Md5_Ctx * ptr_md5Ctx_st, uint8_t *ptr_data, uint32_t dataLen)
{
	crypto_Hash_Status_E ret_md5Stat_en = CRYPTO_HASH_ERROR_NOTSUPPTED;
    
    if(ptr_md5Ctx_st == NULL)
    {
        ret_md5Stat_en = CRYPTO_HASH_ERROR_CTX;
    }
    else if( (ptr_data == NULL) || (dataLen == 0u) )
    {
        ret_md5Stat_en = CRYPTO_HASH_ERROR_INPUTDATA;
    }
    else
    {
        switch(ptr_md5Ctx_st->md5Handler_en)
        {
            case CRYPTO_HANDLER_SW_WOLFCRYPT:
                ret_md5Stat_en = Crypto_Hash_Wc_Md5Update((void*)ptr_md5Ctx_st->arr_md5DataCtx, ptr_data, dataLen);
                break;
            default:
                ret_md5Stat_en = CRYPTO_HASH_ERROR_HDLR;
                break;
        };
    }
	return ret_md5Stat_en;
}

crypto_Hash_Status_E Crypto_Hash_Md5_Final(st_Crypto_Hash_Md5_Ctx * ptr_md5Ctx_st, uint8_t *ptr_digest)
{
   	crypto_Hash_Status_E ret_md5Stat_en = CRYPTO_HASH_ERROR_NOTSUPPTED;
    
    if(ptr_md5Ctx_st == NULL)
    {
        ret_md5Stat_en = CRYPTO_HASH_ERROR_CTX;
    }
    else if(ptr_digest == NULL)
    {      
        ret_md5Stat_en = CRYPTO_HASH_ERROR_OUTPUTDATA;
    }
    else
    {
        switch(ptr_md5Ctx_st->md5Handler_en)
        {
            case CRYPTO_HANDLER_SW_WOLFCRYPT:
                ret_md5Stat_en = Crypto_Hash_Wc_Md5Final((void*)ptr_md5Ctx_st->arr_md5DataCtx, ptr_digest);
                break;
            default:
                ret_md5Stat_en = CRYPTO_HASH_ERROR_HDLR;
                break;
        };
    }
	return ret_md5Stat_en; 
}

static crypto_Hash_Status_E Crypto_Hash_GetHashSize(crypto_Hash_Algo_E hashType_en, uint32_t *hashSize)
{
    crypto_Hash_Status_E ret_val_en = CRYPTO_HASH_SUCCESS;
       
    switch(hashType_en)
    {
        case CRYPTO_HASH_MD5:
            *hashSize = 0x10;   //16 Bytes
            break;
        default:
            ret_val_en = CRYPTO_HASH_ERROR_NOTSUPPTED;
            break;    
    }; 
    return ret_val_en;
}

uint32_t Crypto_Hash_GetHashAndHashSize(crypto_HandlerType_E shaHandler_en, crypto_Hash_Algo_E hashType_en, uint8_t *ptr_wcInputData, 
                                                                                                        uint32_t wcDataLen, uint8_t *ptr_outHash)
{
    crypto_Hash_Status_E hashStatus_en = CRYPTO_HASH_ERROR_FAIL;
    uint32_t hashSize = 0x00;
       
    switch(hashType_en)
    {
        case CRYPTO_HASH_MD5:
            hashStatus_en = Crypto_Hash_Md5_Digest(shaHandler_en, ptr_wcInputData, wcDataLen, ptr_outHash, 1);
            break;
        default:
            hashStatus_en = CRYPTO_HASH_ERROR_NOTSUPPTED;
            break;    
    };
    
    if(hashStatus_en == CRYPTO_HASH_SUCCESS)
    {
        hashStatus_en = Crypto_Hash_GetHashSize(hashType_en, &hashSize);
        
        if(hashStatus_en != CRYPTO_HASH_SUCCESS)
        {
           hashSize = 0x00U;  
        }
    }
    else
    {
       hashSize = 0x00U; 
    }
    return hashSize;
}
