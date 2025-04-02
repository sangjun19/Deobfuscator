/*------------------------------------------------------------------------------
 *
 * Copyright (c) 2011-2025, EURid vzw. All rights reserved.
 * The YADIFA TM software product is provided under the BSD 3-clause license:
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *        * Redistributions of source code must retain the above copyright
 *          notice, this list of conditions and the following disclaimer.
 *        * Redistributions in binary form must reproduce the above copyright
 *          notice, this list of conditions and the following disclaimer in the
 *          documentation and/or other materials provided with the distribution.
 *        * Neither the name of EURid nor the names of its contributors may be
 *          used to endorse or promote products derived from this software
 *          without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 *----------------------------------------------------------------------------*/

/**-----------------------------------------------------------------------------
 * @defgroup
 * @ingroup
 * @brief
 *
 * @{
 *----------------------------------------------------------------------------*/

#include "dnscore/dnscore_config.h"
#include "dnscore/sys_types.h"
#include <unistd.h>
#include <strings.h>
#include <sys/types.h>
#if __unix__
#include <pwd.h>
#include <grp.h>
#endif
#include <sys/stat.h>
#include "dnscore/config_settings.h"
#include "dnscore/cmdline.h"
#include "dnscore/acl_config.h"
#include "dnscore/logger.h"
#include "dnscore/base64.h"
#include "dnscore/u32_treemap.h"
#include "dnscore/parsing.h"
#include "dnscore/tsig.h"
#include "dnscore/file_input_stream.h"
#include "dnscore/bytearray_input_stream.h"

#include "dnscore/config_settings.h"
#include "dnscore/config_file_reader.h"
#include "dnscore/chroot.h"
#include "dnscore/ptr_treemap.h"
#include "dnscore/host_address.h"

#define MODULE_MSG_HANDLE g_system_logger

#define UIDNAME_TAG       0x454d414e444955
#define GIDNAME_TAG       0x454d414e444947

/**
 * This collection links configuration parsing functions to printing functions.
 * Used for extensions (ie: ACL)
 */

static ptr_treemap_t config_section_struct_type_handler_set = PTR_TREEMAP_PTR_EMPTY;
static ptr_treemap_t on_section_read_callback_set = PTR_TREEMAP_ASCIIZ_EMPTY;

/**
 * These two union are used to store functions ptrs as void* in the collections
 * The pointer sizes are supposed to be equal (else dnscore will fail at init)
 *
 * In the event the code has to be compiled on an architecture where void* and void (*f)() are
 * of different size, it will still be time to make a structure that has function pointers for key and value.
 */

union config_set_field_function_as_voidp
{
    config_set_field_function *setter;
    void                      *ptr;
};

typedef union config_set_field_function_as_voidp config_set_field_function_as_voidp;

union config_section_struct_type_handler_as_voidp
{
    config_section_struct_type_handler *handler;
    void                               *ptr;
};

typedef union config_section_struct_type_handler_as_voidp config_section_struct_type_handler_as_voidp;

static const char                                        *config_error_prefix = "config: ";

static u32_treemap_t            section_descriptor_set = U32_TREEMAP_EMPTY;

static uint8_t                  config_current_source = CONFIG_SOURCE_NONE;
static uint8_t                  config_autodefault_after_source = CONFIG_SOURCE_FILE;
static uint8_t                  config_default_source = CONFIG_SOURCE_DEFAULT;
static initialiser_state_t      config_error_codes_init_state = INITIALISE_STATE_INIT;

union code_data_ptr
{
    config_callback_function *function;
    void                     *data;
    intptr_t                  value;
};

typedef union code_data_ptr code_data_ptr;

#if CONFIG_SECTION_DESCRIPTOR_TRACK

#define ADDR2LOC_TAG 0x434f4c3252444441

void config_section_descriptor_file_line_add(config_section_descriptor_t *config_section_descriptor, void *address, const char *filename, int line_number)
{
    address_to_location_t *address_to_location;
    ptr_treemap_node_t    *node = ptr_treemap_insert(&config_section_descriptor->address_to_location_map, address);
    if(node->value == NULL)
    {
        ZALLOC_OBJECT_OR_DIE(address_to_location, address_to_location_t, ADDR2LOC_TAG);
        node->value = address_to_location;
    }
    else
    {
        address_to_location = (address_to_location_t *)node->value;
    }

    string_treemap_node_t *string_node = string_treemap_insert(&config_section_descriptor->location_names_map, filename);
    if(string_node->value == 0)
    {
        string_node->key = strdup(string_node->key);
        string_node->value = 1;
    }

    address_to_location->filename = string_node->key;
    address_to_location->line_number = line_number;
}

static void config_section_descriptor_file_line_clear_address_callback(ptr_treemap_node_t *node)
{
    address_to_location_t *address_to_location = (address_to_location_t *)node->value;
    ZFREE_OBJECT(address_to_location);
}

static void config_section_descriptor_file_line_clear_string_callback(string_treemap_node_t *node) { free((char *)node->key); }

void        config_section_descriptor_file_line_clear(config_section_descriptor_t *config_section_descriptor)
{
    ptr_treemap_callback_and_finalise(&config_section_descriptor->address_to_location_map, config_section_descriptor_file_line_clear_address_callback);
    string_treemap_callback_and_finalise(&config_section_descriptor->location_names_map, config_section_descriptor_file_line_clear_string_callback);
}

address_to_location_t *config_section_descriptor_file_line_get(config_section_descriptor_t *config_section_descriptor, void *address)
{
    ptr_treemap_node_t *node = ptr_treemap_find(&config_section_descriptor->address_to_location_map, address);
    if(node != NULL)
    {
        address_to_location_t *address_to_location = (address_to_location_t *)node->value;
        return address_to_location;
    }
    else
    {
        return NULL;
    }
}

void config_section_descriptor_config_error_update(config_error_t *cfgerr, config_section_descriptor_t *config_section_descriptor, void *address)
{
    if(cfgerr != NULL)
    {
        address_to_location_t *location = config_section_descriptor_file_line_get(config_section_descriptor, address);
        if(location != NULL)
        {
            config_error_set_variable_name(cfgerr, NULL, false);
            cfgerr->line_number = location->line_number;
            cfgerr->has_content = true;
            cfgerr->line[0] = '\0';
            strcpy(cfgerr->file, location->filename);
        }
        else
        {
            cfgerr->line_number = 0;
        }
    }
}

#endif

config_section_descriptor_t *config_section_descriptor_new_instance_ex(const config_section_descriptor_vtbl_s *vtbl, void *data)
{
    config_section_descriptor_t *desc;
    MALLOC_OBJECT_OR_DIE(desc, config_section_descriptor_t, CFGSDESC_TAG);
    desc->base = data;
    desc->vtbl = vtbl;
#if CONFIG_SECTION_DESCRIPTOR_TRACK
    ptr_treemap_init(&desc->address_to_location_map);
    desc->address_to_location_map.compare = ptr_treemap_ptr_node_compare;
    string_treemap_init(&desc->location_names_map);
#endif
    return desc;
}

config_section_descriptor_t *config_section_descriptor_new_instance(const config_section_descriptor_vtbl_s *vtbl)
{
    config_section_descriptor_t *desc;
    desc = config_section_descriptor_new_instance_ex(vtbl, NULL);
    return desc;
}

void config_section_descriptor_delete(config_section_descriptor_t *csd)
{
    config_section_descriptor_file_line_clear(csd);
    free(csd);
}

ya_result config_add_on_section_read_callback(const char *section_name, config_callback_function *on_section_read)
{
    ptr_treemap_node_t *node = ptr_treemap_insert(&on_section_read_callback_set, (char *)section_name);

    if(node->value != NULL)
    {
        return CONFIG_SECTION_CALLBACK_ALREADY_SET; // already exists
    }

    node->key = strdup(section_name);

    code_data_ptr ptr = {.function = on_section_read};

    node->value = ptr.data;

    return SUCCESS;
}

ya_result config_remove_on_section_read_callback(const char *section_name, config_callback_function *on_section_read)
{
    if(section_name == NULL)
    {
        return UNEXPECTED_NULL_ARGUMENT_ERROR;
    }

    ptr_treemap_node_t *node = ptr_treemap_find(&on_section_read_callback_set, section_name);

    if(node != NULL)
    {
        code_data_ptr ptr = {.function = on_section_read};

        if(node->value != ptr.data)
        {
            return CONFIG_SECTION_CALLBACK_NOT_FOUND; // not the right one
        }

        char *key = node->key;
        ptr_treemap_delete(&on_section_read_callback_set, section_name);
        free(key);

        return SUCCESS;
    }
    else
    {
        return CONFIG_SECTION_CALLBACK_NOT_SET;
    }
}

static ya_result config_fire_on_section_read_callback(const char *section_name, int index)
{
    ya_result           return_code = SUCCESS;

    ptr_treemap_node_t *node = ptr_treemap_find(&on_section_read_callback_set, section_name);

    if(node != NULL)
    {
        code_data_ptr             ptr = {.data = node->value};
        config_callback_function *on_section_read = ptr.function;

        if(on_section_read != NULL)
        {
            return_code = on_section_read(section_name, index);
        }
        else
        {
            return_code = CONFIG_SECTION_ERROR;
        }
    }

    return return_code;
}

/** @brief  Yes or No option value parser
 *
 *  @param[in] value in asciiz
 *  @param[out] dest to the value
 *
 *  @retval OK
 *  @retval NOK
 */
ya_result config_set_bool(const char *value, bool *dest, const anytype notused)
{
    (void)notused;

    ya_result ret = parse_bool(value, dest);

    return ret;
}

/** @brief  flag option value parser
 *
 *  @param[in] value in asciiz
 *  @param[out] dest to the value
 *
 *  @retval OK
 *  @retval NOK
 */
ya_result config_set_flag8(const char *value, uint8_t *dest, const anytype mask8)
{
    ya_result return_code;
    bool      b;

    if(ISOK(return_code = config_set_bool(value, &b, mask8)))
    {
        if(b)
        {
            *dest |= mask8._u8;
        }
        else
        {
            *dest &= ~mask8._u8;
        }
    }

    return return_code;
}

/** @brief  flag option value parser
 *
 *  @param[in] value in asciiz
 *  @param[out] dest to the value
 *
 *  @retval OK
 *  @retval NOK
 */
ya_result config_set_flag16(const char *value, uint16_t *dest, const anytype mask16)
{
    ya_result return_code;
    bool      b;

    if(ISOK(return_code = config_set_bool(value, &b, mask16)))
    {
        if(b)
        {
            *dest |= mask16._u16;
        }
        else
        {
            *dest &= ~mask16._u16;
        }
    }

    return return_code;
}

/** @brief  flag option value parser
 *
 *  @param[in] value in asciiz
 *  @param[out] dest to the value
 *
 *  @retval OK
 *  @retval NOK
 */
ya_result config_set_flag32(const char *value, uint32_t *dest, const anytype mask32)
{
    ya_result return_code;
    bool      b;

    if(ISOK(return_code = config_set_bool(value, &b, mask32)))
    {
        if(b)
        {
            *dest |= mask32._u32;
        }
        else
        {
            *dest &= ~mask32._u32;
        }
    }

    return return_code;
}

/** @brief  flag option value parser
 *
 *  @param[in] value in asciiz
 *  @param[out] dest to the value
 *
 *  @retval OK
 *  @retval NOK
 */
ya_result config_set_flag64(const char *value, uint64_t *dest, const anytype mask64)
{
    ya_result return_code;
    bool      b;

    if(ISOK(return_code = config_set_bool(value, &b, mask64)))
    {
        if(b)
        {
            *dest |= mask64._u64;
        }
        else
        {
            *dest &= ~mask64._u64;
        }
    }

    return return_code;
}

/** @brief Integer option value parser
 *
 *  @param[in] value in asciiz
 *  @param[out] dest to the value
 *
 *  @retval OK
 *  @retval NOK
 */

ya_result config_set_u64(const char *value, uint64_t *dest, const anytype notused)
{
    (void)notused;

    errno = 0;
    while(isspace(*value) && (*value != '\0'))
    {
        ++value;
    }
    if(*value == '-')
    {
        return CONFIG_VALUE_OUT_OF_RANGE;
    }
    char    *first_invalid = NULL;
    uint64_t tmp = strtoull(value, &first_invalid, 10);
    int      err = errno;
    if(err == 0)
    {
        if(first_invalid != value)
        {
            *dest = tmp;
            return SUCCESS;
        }
        else
        {
            return PARSE_ERROR;
        }
    }
    else
    {
        return MAKE_ERRNO_ERROR(err);
    }
}

/** @brief Integer option value parser
 *
 *  @param[in] value in asciiz
 *  @param[out] dest to the value
 *
 *  @retval OK
 *  @retval NOK
 */

ya_result config_set_u32(const char *value, uint32_t *dest, const anytype notused)
{
    (void)notused;

    errno = 0;
    while(isspace(*value) && (*value != '\0'))
    {
        ++value;
    }
    if(*value == '-')
    {
        return CONFIG_VALUE_OUT_OF_RANGE;
    }
    char   *first_invalid = NULL;
    int64_t tmp = strtoull(value, &first_invalid, 10);
    int     err = errno;
    if(err == 0)
    {
        if(first_invalid != value)
        {
            if((tmp >= 0) && (tmp <= UINT32_MAX))
            {
                *dest = tmp;
                return SUCCESS;
            }
            else
            {
                return MAKE_ERRNO_ERROR(EOVERFLOW);
            }
        }
        else
        {
            return PARSE_ERROR;
        }
    }
    else
    {
        return MAKE_ERRNO_ERROR(err);
    }
}

/** @brief Integer option value parser
 *
 *  @param[in] value in asciiz
 *  @param[out] dest to the value
 *
 *  @retval OK
 *  @retval NOK
 */

ya_result config_set_s32(const char *value, int32_t *dest, const anytype notused)
{
    (void)notused;

    errno = 0;
    while(isspace(*value) && (*value != '\0'))
    {
        ++value;
    }
    char   *first_invalid = NULL;
    int64_t tmp = strtoll(value, &first_invalid, 10);
    int     err = errno;
    if(err == 0)
    {
        if(first_invalid != value)
        {
            if((tmp >= INT32_MIN) && (tmp <= INT32_MAX))
            {
                *dest = (int32_t)tmp;
                return SUCCESS;
            }
            else
            {
                return MAKE_ERRNO_ERROR(EOVERFLOW);
            }
        }
        else
        {
            return PARSE_ERROR;
        }
    }
    else
    {
        return MAKE_ERRNO_ERROR(err);
    }
}

/** @brief Integer option value parser
 *
 *  @param[in] value in asciiz
 *  @param[out] dest to the value
 *
 *  @retval OK
 *  @retval NOK
 */

ya_result config_set_u32_range(const char *value, uint32_t *dest, const anytype range)
{
    errno = 0;
    while(isspace(*value) && (*value != '\0'))
    {
        ++value;
    }
    if(*value == '-')
    {
        return CONFIG_VALUE_OUT_OF_RANGE;
    }
    char   *first_invalid = NULL;
    int64_t tmp = strtoll(value, &first_invalid, 10);
    int     err = errno;
    if(err == 0)
    {
        if(first_invalid != value)
        {
            if(tmp >= range._2u32[0] && tmp <= range._2u32[1])
            {
                *dest = (uint32_t)tmp;
                return SUCCESS;
            }
            else
            {
                return CONFIG_VALUE_OUT_OF_RANGE;
            }
        }
        else
        {
            return PARSE_ERROR;
        }
    }
    else
    {
        return MAKE_ERRNO_ERROR(err);
    }
}

/** @brief Integer option value parser
 *
 *  @param[in] value in asciiz
 *  @param[out] dest to the value
 *
 *  @retval OK
 *  @retval NOK
 */

ya_result config_set_u32_clamp(const char *value, uint32_t *dest, const anytype range)
{
    errno = 0;
    while(isspace(*value) && (*value != '\0'))
    {
        ++value;
    }
    char   *first_invalid = NULL;
    int64_t tmp = strtoll(value, &first_invalid, 10);
    int     err = errno;
    if(err == 0)
    {
        if(first_invalid != value)
        {
            if(tmp < range._2u32[0])
            {
                tmp = range._2u32[0];
            }
            else if(tmp > range._2u32[1])
            {
                tmp = range._2u32[1];
            }

            *dest = (uint32_t)tmp;

            return SUCCESS;
        }
        else
        {
            return PARSE_ERROR;
        }
    }
    else
    {
        return MAKE_ERRNO_ERROR(err);
    }
}

/** @brief Integer option value parser
 *
 *  @param[in] value in asciiz
 *  @param[out] dest to the value
 *
 *  @retval OK
 *  @retval NOK
 */

ya_result config_set_u16(const char *value, uint16_t *dest, const anytype notused)
{
    (void)notused;

    errno = 0;
    while(isspace(*value) && (*value != '\0'))
    {
        ++value;
    }
    if(*value == '-')
    {
        return CONFIG_VALUE_OUT_OF_RANGE;
    }
    char   *first_invalid = NULL;
    int64_t tmp = strtoll(value, &first_invalid, 10);
    int     err = errno;
    if(err == 0)
    {
        if(first_invalid != value)
        {
            if((tmp >= 0) && (tmp <= U16_MAX))
            {
                *dest = (uint16_t)tmp;
                return SUCCESS;
            }
            else
            {
                return MAKE_ERRNO_ERROR(EOVERFLOW);
            }
        }
        else
        {
            return PARSE_ERROR;
        }
    }
    else
    {
        return MAKE_ERRNO_ERROR(err);
    }
}

ya_result config_set_u8(const char *value, uint8_t *dest, const anytype notused)
{
    (void)notused;

    errno = 0;
    char   *first_invalid = NULL;
    int64_t tmp = strtoll(value, &first_invalid, 10);
    int     err = errno;
    if(err == 0)
    {
        if(first_invalid != value)
        {
            if((tmp >= 0) && (tmp <= U8_MAX))
            {
                *dest = (uint8_t)tmp;
                return SUCCESS;
            }
            else
            {
                return MAKE_ERRNO_ERROR(EOVERFLOW);
            }
        }
        else
        {
            return PARSE_ERROR;
        }
    }
    else
    {
        return MAKE_ERRNO_ERROR(err);
    }
}

ya_result config_inc_u8(const char *value_notused, uint8_t *dest, const anytype notused)
{
    (void)value_notused;
    (void)notused;

    (*dest)++;

    return OK;
}

ya_result config_set_dnskey_algorithm(const char *value, uint8_t *dest, const anytype notused)
{
    (void)notused;

    ya_result ret;
    uint32_t  val;
    if(ISOK(ret = parse_u32_check_range(value, &val, 1, 255, BASE_10)))
    {
        *dest = (uint8_t)val;
    }
    else
    {
        ret = dns_encryption_algorithm_from_case_name(value, dest);
    }

    return ret;
}

/** @brief String parser
 *
 *  @param[in] value
 *  @param[in] config_command
 *  @param[out] config
 *
 *  @retval OK
 *  @retval NOK
 */

ya_result config_set_string(const char *value, char **dest, const anytype notused)
{
    (void)notused;

    if(*dest != NULL)
    {
        if(strcmp(*dest, value) != 0)
        {
            free(*dest);
            *dest = strdup(value);
        }
    }
    else
    {
        *dest = strdup(value);
    }

    return OK;
}

/** @brief String parser
 *
 *  @param[in] value
 *  @param[in] config_command
 *  @param[out] config
 *
 *  @retval OK
 *  @retval NOK
 */

ya_result config_set_string_copy(const char *value, char *dest, const anytype maxlen)
{
    size_t len = strlen(value);
    if(len > maxlen._u32 - 1)
    {
        return CONFIG_TEXT_LENGTH_TOO_BIG;
    }

    memcpy(dest, value, len);
    dest[len] = '\0';

    return len + 1;
}

/** @brief String parser
 *
 *  @param[in] value
 *  @param[in] config_command
 *  @param[out] config
 *
 *  @retval OK
 *  @retval NOK
 */

ya_result config_append_string_array_item(const char *value, ptr_vector_t *dest, const anytype maxsize)
{
    if(ptr_vector_size(dest) >= maxsize._s32)
    {
        return CONFIG_ARRAY_SIZE_TOO_BIG;
    }

    if(value != NULL)
    {
        ptr_vector_append(dest, strdup(value));
    }

    return ptr_vector_size(dest);
}

/** @brief Password parser
 *
 *  @param[in] value
 *  @param[in] config_command
 *  @param[out] config
 *
 *  @retval OK
 *  @retval NOK
 */

ya_result config_set_password(const char *value, char **destp, const anytype notused)
{
    (void)notused;

    if(*destp != NULL)
    {
        if(strcmp(*destp, value) != 0)
        {
            size_t n = strlen(*destp);

            for(size_t i = 0; i < n; i++)
            {
                (*destp)[i] = rand();
            }

            free(*destp);
            *destp = strdup(value);
        }
    }
    else
    {
        *destp = strdup(value);
    }

    return SUCCESS;
}

/** @brief String parser
 *
 *  @param[in] value
 *  @param[in] config_command
 *  @param[out] config
 *
 *  @retval OK
 *  @retval NOK
 */

ya_result config_set_fqdn(const char *value, uint8_t **dest, const anytype notused)
{
    (void)notused;

    ya_result return_value;

    uint8_t   tmp[DOMAIN_LENGTH_MAX];

    return_value = dnsname_init_with_cstr(tmp, value);

    if(ISOK(return_value))
    {
        if(*dest != NULL)
        {
            if(!dnsname_equals(*dest, tmp))
            {
                free(*dest);
                *dest = dnsname_dup(tmp);
            }
        }
        else
        {
            *dest = dnsname_dup(tmp);
        }
    }

    return return_value;
}

/** @brief Path parser
 *
 *  Ensures that the stored value ends with '/'
 *
 *  @param[in] value
 *  @param[in] config_command
 *  @param[out] config
 *
 *  @retval OK
 *  @retval NOK
 */

ya_result config_set_path(const char *value, char **dest, const anytype notused)
{
    (void)notused;

    size_t len = strlen(value);

    if(*dest != NULL)
    {
        size_t dest_len = strlen(*dest);

        if(value[len - 1] != '/')
        {
            if(dest_len == len + 1) // implies last char of *dest is '/'
            {
                if(memcmp(*dest, value, len) == 0)
                {
                    return SUCCESS;
                }
            }
        }
        else
        {
            if(strcmp(*dest, value) == 0)
            {
                return SUCCESS;
            }
        }

        free(*dest);
        *dest = NULL;
    }

    if(value[len - 1] != '/')
    {
        char *tmp = (char *)malloc(len + 2);
        if(tmp != NULL)
        {
            memcpy(tmp, value, len);
            tmp[len] = '/';
            tmp[len + 1] = '\0';
            *dest = tmp;
        }
        else
        {
            return MAKE_ERRNO_ERROR(ENOMEM);
        }
    }
    else
    {
        if((*dest = strdup(value)) == NULL)
        {
            return MAKE_ERRNO_ERROR(ENOMEM);
        }
    }

    return SUCCESS;
}

ya_result config_set_logpath(const char *value, char **dest, const anytype notused)
{
    ya_result return_code;

    if(ISOK(return_code = config_set_path(value, dest, notused)))
    {
        config_set_log_base_path(*dest);
    }

    return return_code;
}

ya_result config_set_chroot(const char *value, char **dest, const anytype notused)
{
    (void)notused;

#if __unix__
    ya_result return_code;

    if(ISOK(return_code = config_set_path(value, dest, notused)))
    {
        return_code = chroot_set_path(*dest);
        // chdir(*dest);
    }

    return return_code;
#else
    return FEATURE_NOT_IMPLEMENTED_ERROR;
#endif
}

ya_result config_set_file(const char *value, char **dest, const anytype notused)
{
    (void)notused;

    struct stat fileinfo;

    if(filestat(value, &fileinfo) < 0)
    {
        int err = ERRNO_ERROR;
        if(err == MAKE_ERRNO_ERROR(ENOENT))
        {
            return CONFIG_FILE_NOT_FOUND;
        }
        else
        {
            return ERRNO_ERROR;
        }
    }
    /* Is it a regular file */
    if(!S_ISREG(fileinfo.st_mode))
    {
        return CONFIG_NOT_A_REGULAR_FILE;
    }

    if(*dest != NULL)
    {
        if(strcmp(*dest, value) != 0)
        {
            free(*dest);
            *dest = strdup(value);
        }
    }
    else
    {
        *dest = strdup(value);
    }

    return SUCCESS;
}

/** @brief UID parser
 *
 *  @param[in] value
 *  @param[in] config_command
 *  @param[out] config
 *
 *  @retval OK
 *  @retval CONFIG_BAD_UID
 */

ya_result config_set_uid_t(const char *value, uid_t *dest, const anytype notused)
{
    (void)notused;
#if __unix__

    if((*value == '\0') || (strcmp(value, "-") == 0))
    {
        *dest = getuid();
    }
    else
    {
        struct passwd  pwd;
        struct passwd *result;
        char          *buffer;

        long           buffer_size = sysconf(_SC_GETPW_R_SIZE_MAX);

        /*
         * This fix has been made for FreeBSD that returns -1 for the above call
         */

        if(buffer_size < 0)
        {
            buffer_size = 1024;
        }

        MALLOC_OR_DIE(char *, buffer, buffer_size, UIDNAME_TAG);
        getpwnam_r(value, &pwd, buffer, buffer_size, &result);
        *dest = pwd.pw_uid;
        free(buffer);

        if(result == NULL)
        {
            uint32_t val;
            if(FAIL(parse_u32_check_range(value, &val, 0, U32_MAX, BASE_10)))
            {
                return CONFIG_BAD_UID;
            }
            *dest = val;
        }
    }
#endif

    return SUCCESS;
}

/** @brief GID parser
 *
 *  @param[in] value
 *  @param[in] config_command
 *  @param[out] config
 *
 *  @retval OK
 *  @retval CONFIG_BAD_GID
 */

ya_result config_set_gid_t(const char *value, gid_t *dest, const anytype notused)
{
#if __unix__
    (void)notused;

    if((*value == '\0') || (strcmp(value, "-") == 0))
    {
        *dest = getgid();
    }
    else
    {
        struct group  grp;
        struct group *result;
        char         *buffer;

        long          buffer_size = sysconf(_SC_GETGR_R_SIZE_MAX);

        /*
         * This fix has been made for FreeBSD that returns -1 for the above call
         */

        if(buffer_size < 0)
        {
            buffer_size = 1024;
        }

        MALLOC_OR_DIE(char *, buffer, buffer_size, GIDNAME_TAG);

        getgrnam_r(value, &grp, buffer, buffer_size, &result);
        *dest = grp.gr_gid;
        free(buffer);

        if(result == NULL)
        {
            uint32_t val;

            if(FAIL(parse_u32_check_range(value, &val, 0, U32_MAX, BASE_10)))
            {
                return CONFIG_BAD_GID;
            }

            *dest = val;
        }
    }
#endif

    return SUCCESS;
}

ya_result config_set_dnsclass(const char *value, uint16_t *dest, const anytype notused)
{
    (void)notused;

    ya_result ret;
    uint16_t  qclass;

    if(FAIL(ret = dns_class_from_case_name(value, &qclass)))
    {
        return ret;
    }

    *dest = ntohs(qclass);

    return SUCCESS;
}

ya_result config_set_dnstype(const char *value, uint16_t *dest, const anytype notused)
{
    (void)notused;

    ya_result ret;
    uint16_t  rtype;

    if(FAIL(ret = dns_type_from_case_name(value, &rtype)))
    {
        return ret;
    }

    *dest = ntohs(rtype);

    return SUCCESS;
}

ya_result config_set_enum_value(const char *value, uint32_t *dest, const anytype enum_value_name_table)
{
    ya_result           return_code;
    uint32_t            integer_value;

    value_name_table_t *table = (value_name_table_t *)enum_value_name_table._voidp;

    if(ISOK(return_code = value_name_table_get_value_from_casename(table, value, &integer_value)))
    {
        *dest = integer_value;
    }

    return return_code;
}

ya_result config_set_enum8_value(const char *value, uint8_t *dest, const anytype enum_value_name_table)
{
    ya_result           return_code;
    uint32_t            integer_value;

    value_name_table_t *table = (value_name_table_t *)enum_value_name_table._voidp;

    if(ISOK(return_code = value_name_table_get_value_from_casename(table, value, &integer_value)))
    {
        *dest = integer_value;
    }

    return return_code;
}

/*
 * IP port n,
 */
ya_result config_set_host_list(const char *value, host_address_t **dest, const anytype settings)
{
    if(value == NULL) /* nothing to do */
    {
        return UNEXPECTED_NULL_ARGUMENT_ERROR;
    }

    ya_result   return_code;
    const char *from = value;
    uint16_t    ip_port = 0;
#if DNSCORE_HAS_TSIG_SUPPORT
    tsig_key_t *tsig = NULL;
#endif
    uint8_t ip_size;
    uint8_t host_type = HOST_ADDRESS_NONE;

    bool    eol = (*from == '\0');

    uint8_t flags = settings._8u8[0];
    uint8_t tls = HOST_ADDRESS_TLS_NOT_SET;
    uint8_t ip_buffer[DOMAIN_LENGTH_MAX];
    uint8_t total_allowed = settings._8u8[1];

    if(!(flags & CONFIG_HOST_LIST_FLAGS_APPEND))
    {
        /* delete the content of the list */
        if(*dest != NULL)
        {
            host_address_delete_list(*dest);
            *dest = NULL;
        }
    }
    else
    {
        /* find the last node of the list so the new ones will be appended */
        uint32_t counter = 0;
        while(*dest != NULL)
        {
            counter++;
            dest = &(*dest)->next;
        }

        if(counter > total_allowed)
        {
            return CONFIG_TOO_MANY_HOSTS;
        }

        total_allowed -= counter;
    }

    while(!eol)
    {
        /* skip the white spaces */

        from = (char *)parse_skip_spaces(from);

        const char *to = from;

        /* get the end of statement */

        to = (char *)parse_skip_until_chars(to, ",;", 2);

        if(to == from)
        {
            /* No new statement */
            break;
        }

        eol = (*to == '\0');

        /* now skip from until space */

        const char *port_or_key_or_tls = from;

        port_or_key_or_tls = (char *)parse_next_space(port_or_key_or_tls);

        const char *next_word = port_or_key_or_tls;

        next_word = parse_skip_spaces(next_word);

        bool ip_only = (next_word >= to);

        port_or_key_or_tls = MIN(port_or_key_or_tls, to);

        host_type = HOST_ADDRESS_NONE;

        if(FAIL(return_code = parse_ip_address(from, port_or_key_or_tls - from, ip_buffer, sizeof(ip_buffer))))
        {
            if(!(flags & CONFIG_HOST_LIST_FLAGS_FQDN))
            {
                return CONFIG_FQDN_NOT_ALLOWED;
            }

            if(FAIL(return_code = dnsname_init_check_nostar_with_charp(ip_buffer, from, port_or_key_or_tls - from)))
            {
                return return_code;
            }

            host_type = HOST_ADDRESS_DNAME;
        }

        ip_size = (uint8_t)return_code;

        if(ip_size == 4)
        {
            if(!(flags & CONFIG_HOST_LIST_FLAGS_IPV4))
            {
                return CONFIG_IPV4_NOT_ALLOWED;
            }
            host_type = HOST_ADDRESS_IPV4;
        }

        if(ip_size == 16)
        {
            if(!(flags & CONFIG_HOST_LIST_FLAGS_IPV6))
            {
                return CONFIG_IPV6_NOT_ALLOWED;
            }
            host_type = HOST_ADDRESS_IPV6;
        }

        ip_port = 0;

#if DNSCORE_HAS_TSIG_SUPPORT
        tsig = NULL;
#endif
        if(!ip_only)
        {
            /* parse & skip 'port */

            uint8_t key_dnsname[DOMAIN_LENGTH_MAX + 1];
#if DNSCORE_HAS_TSIG_SUPPORT
            char key_name[DOMAIN_TEXT_BUFFER_SIZE];
#endif
            static const char *keywords[4] = {"port", "key", "notls", "tls"};

            for(;;)
            {
                if(to - port_or_key_or_tls == 0)
                {
                    break;
                }

                int32_t index;
                return_code = parse_skip_word_specific(port_or_key_or_tls, to - port_or_key_or_tls, keywords, 4, &index);

                if(FAIL(return_code))
                {
                    // bad keyword
                    log_err("%sgarbage around '%s'", config_error_prefix, port_or_key_or_tls);

                    /* parse error, expected something */

                    return CONFIG_KEY_PARSE_ERROR;
                }

                switch(index)
                {
                    case 0: // port
                    {
                        /// get PORT
                        if(!(flags & CONFIG_HOST_LIST_FLAGS_PORT))
                        {
                            return CONFIG_PORT_NOT_ALLOWED;
                        }

                        next_word = parse_skip_spaces(&port_or_key_or_tls[return_code]);

                        uint32_t port_value;

                        if(FAIL(return_code = parse_u32_check_range(next_word, &port_value, 1, U16_MAX, BASE_10)))
                        {
                            /* parse error, expected something */

                            log_err("%sport parse error around '%s'", config_error_prefix, next_word);

                            return CONFIG_EXPECTED_VALID_PORT_VALUE;
                        }

                        next_word = parse_next_space(next_word);
                        next_word = MIN(next_word, to);

                        port_or_key_or_tls = next_word;

                        ip_port = (uint16_t)port_value;

                        break;
                    }
                    case 1: // key
                    {
                        /// get KEY
#if DNSCORE_HAS_TSIG_SUPPORT
                        if(!(flags & CONFIG_HOST_LIST_FLAGS_TSIG))
                        {
                            return CONFIG_TSIG_NOT_ALLOWED;
                        }

                        const char *key_name_start = parse_skip_spaces(&port_or_key_or_tls[return_code]);

                        key_name_start = (char *)parse_skip_spaces(key_name_start);

                        next_word = key_name_start;

                        next_word = (char *)parse_next_space(next_word);
                        next_word = MIN(next_word, to);

                        // port_or_key = next_word;

                        size_t key_name_len = next_word - key_name_start;

                        if(key_name_len < DOMAIN_TEXT_LENGTH_MAX)
                        {
                            memcpy(key_name, key_name_start, key_name_len);

                            key_name[key_name_len] = '\0';

                            //*next_word++ = '\0';
                            port_or_key_or_tls = next_word;

                            if(ISOK(return_code = dnsname_init_check_star_with_cstr(key_dnsname, key_name)))
                            {
                                tsig = tsig_get(key_dnsname);

                                if(tsig == NULL)
                                {
                                    log_err("%skey '%s' has not been defined", config_error_prefix, key_name);

                                    return CONFIG_KEY_UNKNOWN;
                                }
                            }
                            else
                            {
                                log_err("%skey name parse error around '%s': %r", config_error_prefix, key_name, return_code);

                                return CONFIG_KEY_PARSE_ERROR;
                            }
                        }
                        else
                        {
                            log_err("%skey name is too big", config_error_prefix);

                            return CONFIG_KEY_PARSE_ERROR;
                        }

                        break;
#else
                        return CONFIG_TSIG_NOT_ALLOWED; // no TSIG support ...
#endif
                    }
                    case 2: // notls
                    {
                        tls = HOST_ADDRESS_TLS_DISABLE;

                        next_word = parse_skip_spaces(&port_or_key_or_tls[return_code]);
                        next_word = MIN(next_word, to);
                        port_or_key_or_tls = next_word;
                        break;
                    }
                    case 3: // tls
                    {
                        tls = HOST_ADDRESS_TLS_ENFORCE;

                        next_word = parse_skip_spaces(&port_or_key_or_tls[return_code]);
                        next_word = MIN(next_word, to);
                        port_or_key_or_tls = next_word;
                        break;
                    }
                } // switch
            } // for
        }

        // Only testing here because before until know I don't know if what was parsed was empty or not

        if(total_allowed == 0)
        {
            return CONFIG_TOO_MANY_HOSTS;
        }

        --total_allowed;

        // Now we can add a host structure node

        host_address_t *address;

        address = host_address_new_instance(); // sets version=0, next = NULL, tsig = NULL

#if DNSCORE_HAS_TSIG_SUPPORT
        address->tsig = tsig;
#endif

        switch(host_type)
        {
            case HOST_ADDRESS_IPV4:
            {
                host_address_set_ipv4(address, ip_buffer, htons(ip_port));

                break;
            }
            case HOST_ADDRESS_IPV6:
            {
                host_address_set_ipv6(address, ip_buffer, htons(ip_port));

                break;
            }
            case HOST_ADDRESS_DNAME:
            {
                host_address_set_dname(address, ip_buffer, htons(ip_port));

                break;
            }
            case HOST_ADDRESS_NONE:
            {
                host_address_delete(address);
                return PARSEIP_ERROR;
            }
        }

        address->tls = tls;

        *dest = address;
        dest = &address->next;

        from = to + 1;
    }

    return SUCCESS;
}

/*
 * obfuscated format: something used by the server to store information
 *     that it needs as is.  (ie: dynamic provisioning related fields)
 *
 * 01 83 [flags] [serial] [checksum]
 */

ya_result config_set_bytes(const char *value, void *dest, const anytype sizeoftarget)
{
    (void)dest;
    (void)sizeoftarget;

    ya_result return_value;

    return_value = base64_decode(value, /*sizeoftarget._u32*/ strlen(value), (uint8_t *)dest);

    return return_value;
}

/**
 * [hmac:]name:key
 */

ya_result config_set_tsig_key(const char *value, tsig_key_t **dest, const anytype unused)
{
    (void)unused;
    ya_result   ret;
    char       *token2;
    char       *token3;
    const char *value_limit = &value[strlen(value)];

    token2 = strchr(value, ':');
    if(token2 == NULL)
    {
        return PARSE_INVALID_ARGUMENT; // expects at least one ':'
    }

    token3 = strchr(token2 + 1, ':');

    size_t  hmac_name_size;
    size_t  key_name_size;
    char    hmac_name[32];
    uint8_t key_name[DOMAIN_LENGTH_MAX];
    uint8_t key_bytes[1024];

    if(token3 == NULL)
    {
        // hmac-md5 OR hmac-sha256 :name:key
        memcpy(hmac_name, "hmac-md5", 9);
    }
    else
    {
        // hmac:name:key

        hmac_name_size = token2 - value;

        if(hmac_name_size < sizeof(hmac_name) - 1)
        {
            memcpy(hmac_name, value, hmac_name_size);
            hmac_name[hmac_name_size] = '\0';
        }
        else
        {
            return BUFFER_WOULD_OVERFLOW;
        }

        value = token2 + 1;
        token2 = token3;
    }

    key_name_size = token2 - value;

    ret = dnsname_init_with_charp_locase(key_name, value, key_name_size);

    if(ISOK(ret))
    {
        ++token2;
        size_t key_size = value_limit - token2;

        if(BASE64_DECODED_SIZE(key_size) <= sizeof(key_bytes))
        {
            ret = base64_decode(token2, key_size, key_bytes);

            if(ISOK(ret))
            {
                key_size = (size_t)ret;

                if(ISOK(ret = tsig_get_hmac_algorithm_from_friendly_name(hmac_name)))
                {
                    uint32_t hmac_algorithm = (uint32_t)ret;

                    ret = tsig_register(key_name, key_bytes, key_size, hmac_algorithm);

                    if(ISOK(ret))
                    {
                        tsig_key_t *tsig_key = tsig_get(key_name);
                        if(dest != NULL)
                        {
                            *dest = tsig_key;
                        }
                    }

                    return ret;
                }
                else
                {
                    return UNKNOWN_NAME;
                }
            }
            else
            {
                return ret;
            }
        }
        else
        {
            return BUFFER_WOULD_OVERFLOW;
        }
    }
    else
    {
        return ret;
    }
}

ya_result config_set_obsolete(const char *value, void *dest, const anytype notused)
{
    (void)dest;
    (void)notused;

    if(logger_is_running())
    {
        log_warn("parameter '%s' is not used", value);
    }
    else
    {
        osformatln(termerr, "parameter '%s' is not used", value);
    }

    return SUCCESS;
}

void config_init_error_codes()
{
    if(initialise_state_begin(&config_error_codes_init_state))
    {
        error_register(CONFIG_SECTION_ALREADY_REGISTERED, "CONFIG_SECTION_ALREADY_REGISTERED");
        error_register(CONFIG_ALIAS_CHAIN_TOO_BIG, "CONFIG_ALIAS_CHAIN_TOO_BIG");

        error_register(CONFIG_PARSE_SECTION_TAG_NOT_CLOSED, "CONFIG_PARSE_SECTION_TAG_NOT_CLOSED");
        error_register(CONFIG_PARSE_UNEXPECTED_SECTION_OPEN, "CONFIG_PARSE_UNEXPECTED_SECTION_OPEN");
        error_register(CONFIG_PARSE_UNEXPECTED_SECTION_CLOSE, "CONFIG_PARSE_UNEXPECTED_SECTION_CLOSE");
        error_register(CONFIG_PARSE_CLOSED_WRONG_SECTION, "CONFIG_PARSE_CLOSED_WRONG_SECTION");
        error_register(CONFIG_PARSE_SECTION_TAG_TOO_SMALL, "CONFIG_PARSE_SECTION_TAG_TOO_SMALL");
        error_register(CONFIG_PARSE_INCLUDE_EXPECTED_FILE_PATH, "CONFIG_PARSE_INCLUDE_EXPECTED_FILE_PATH");
        error_register(CONFIG_PARSE_UNKNOWN_KEYWORD, "CONFIG_PARSE_UNKNOWN_KEYWORD");
        error_register(CONFIG_PARSE_EXPECTED_VALUE, "CONFIG_PARSE_EXPECTED_VALUE");

        error_register(CONFIG_UNKNOWN_SETTING, "CONFIG_UNKNOWN_SETTING");
        error_register(CONFIG_VALUE_OUT_OF_RANGE, "CONFIG_VALUE_OUT_OF_RANGE");
        error_register(CONFIG_FILE_PATH_TOO_BIG, "CONFIG_FILE_PATH_TOO_BIG");
        error_register(CONFIG_BAD_UID, "CONFIG_BAD_UID");
        error_register(CONFIG_BAD_GID, "CONFIG_BAD_GID");

        error_register(CONFIG_TEXT_LENGTH_TOO_BIG, "CONFIG_TEXT_LENGTH_TOO_BIG");
        error_register(CONFIG_ARRAY_SIZE_TOO_BIG, "CONFIG_ARRAY_SIZE_TOO_BIG");

        error_register(CONFIG_LOGGER_HANDLE_ALREADY_DEFINED, "CONFIG_LOGGER_HANDLE_ALREADY_DEFINED");
        error_register(CONFIG_LOGGER_INVALID_DEBUGLEVEL, "CONFIG_LOGGER_INVALID_DEBUGLEVEL");

        error_register(CONFIG_KEY_INCOMPLETE_KEY, "CONFIG_KEY_INCOMPLETE_KEY");
        error_register(CONFIG_KEY_UNSUPPORTED_ALGORITHM, "CONFIG_KEY_UNSUPPORTED_ALGORITHM");

        initialise_state_ready(&config_error_codes_init_state);
    }
}

uint8_t config_get_source() { return config_current_source; }

void    config_set_source(uint8_t l) { config_current_source = l; }

/**
 * If the source level has been parsed, automatically fill the default values
 * for fields that are not set yet.
 *
 * Default: CONFIG_SOURCE_FILE = 128
 *
 * @param l after what level do we automatically set the default values in the container ?
 */

void config_set_autodefault_after_source(uint8_t l) { config_autodefault_after_source = l; }

/**
 * If the source level has been parsed, automatically fill the default values
 * for fields that are not set yet.
 *
 * @return after what level do we automatically set the default values in the container
 */

uint8_t   config_get_autodefault_after_source() { return config_autodefault_after_source; }

uint8_t   config_get_default_source() { return config_default_source; }

void      config_set_default_source(uint8_t l) { config_default_source = l; }

ya_result config_init()
{
    config_init_error_codes();
    return SUCCESS;
}

/**
 *
 * Will store the pointer to the descriptor into a global collection.
 * Returns an error if the node exists already and is not identical.
 * Which should lead to free the redundant descriptor by the caller.
 *
 * @param section_descriptor
 * @param priority
 * @return
 */

ya_result config_register(config_section_descriptor_t *section_descriptor, int32_t priority)
{
    if(priority < 0)
    {
        priority = 0x1000;
    }

    u32_treemap_node_t *node = u32_treemap_find(&section_descriptor_set, (uint32_t)priority);
    if(node == NULL)
    {
        node = u32_treemap_insert(&section_descriptor_set, (uint32_t)priority);

        node->value = /*(void*)*/ section_descriptor;

        return SUCCESS;
    }
    else
    {
        if(node->value == section_descriptor)
        {
            return SUCCESS;
        }
        else
        {
            return CONFIG_SECTION_ALREADY_REGISTERED;
        }
    }
}

/**
 * Registers a descriptor at the given priority
 *
 * @param section_descritor config descriptor
 * @param priority config priority
 *
 * @return an error code
 */

ya_result config_register_const(const config_section_descriptor_t *section_descriptor, int32_t priority)
{
    ya_result ret;
    ret = config_register((config_section_descriptor_t *)section_descriptor, priority);
    return ret;
}

/**
 * Removes the registration of a descriptor
 *
 * @param section_descritor config descriptor
 *
 * @return an error code
 */

ya_result config_unregister(config_section_descriptor_t *section_descriptor)
{
    u32_treemap_iterator_t iter;
    u32_treemap_iterator_init(&section_descriptor_set, &iter);
    while(u32_treemap_iterator_hasnext(&iter))
    {
        u32_treemap_node_t *node = u32_treemap_iterator_next_node(&iter);
        if(node->value == section_descriptor)
        {
            u32_treemap_delete(&section_descriptor_set, node->key);
            return SUCCESS;
        }
    }

    return ERROR;
}

/**
 * Removes the registration of a descriptor identified by its name
 *
 * @param name the config descriptor name
 *
 * @return an error code
 */

config_section_descriptor_t *config_unregister_by_name(const char *name)
{
    u32_treemap_iterator_t iter;
    u32_treemap_iterator_init(&section_descriptor_set, &iter);
    while(u32_treemap_iterator_hasnext(&iter))
    {
        u32_treemap_node_t *node = u32_treemap_iterator_next_node(&iter);
        if(node->value != NULL)
        {
            config_section_descriptor_t *desc = (config_section_descriptor_t *)node->value;
            if(desc->vtbl != NULL)
            {
                if(strcmp(desc->vtbl->name, name) == 0)
                {
                    u32_treemap_delete(&section_descriptor_set, node->key);
                    return desc;
                }
            }
        }
    }
    return NULL;
}

ya_result config_set_section_default(config_section_descriptor_t *section_descriptor, config_error_t *cfgerr)
{
    int32_t err = SUCCESS;

    if((section_descriptor->vtbl->table != NULL) && (section_descriptor->base != NULL))
    {
        section_descriptor->vtbl->init(section_descriptor);

        for(const config_table_descriptor_item_t *item = section_descriptor->vtbl->table; item->name != NULL; item++)
        {
            if((item->default_value_string != NULL) && (item->setter != NULL))
            {
                if(FAIL(err = config_value_set(section_descriptor, item->name, item->default_value_string, cfgerr)))
                {
                    if((cfgerr != NULL) && !cfgerr->has_content)
                    {
                        config_error_set_variable_name(cfgerr, (char *)item->name, false);
                        cfgerr->line_number = 0;
                        cfgerr->has_content = true;
                        strcpy_ex(cfgerr->file, "default values", sizeof(cfgerr->file));
                        strcpy_ex(cfgerr->line, item->name, sizeof(cfgerr->line));
                        snformat(cfgerr->line, sizeof(cfgerr->line), "%s \"%s\"", item->name, item->default_value_string);
                    }

                    break;
                }
            }
        }
    }

    return err;
}

ya_result config_read_section(const char *the_configuration_file_path, config_error_t *cfgerr, const char *section_name)
{
    ya_result err = SUCCESS;

    char      configuration_file_path[PATH_MAX];

    // if the passed value is a pointer into a configuration structure,
    // there is a risk that the value is freed and replaced by a different one
    // => bad
    // so a copy is done first

    strcpy_ex(configuration_file_path, the_configuration_file_path, sizeof(configuration_file_path));

    u32_treemap_iterator_t iter;
    u32_treemap_iterator_init(&section_descriptor_set, &iter);
    while(u32_treemap_iterator_hasnext(&iter))
    {
        u32_treemap_node_t          *node = u32_treemap_iterator_next_node(&iter);
        config_section_descriptor_t *section_descriptor = (config_section_descriptor_t *)node->value;

        if((section_name != NULL) && (strcmp(section_descriptor->vtbl->name, section_name) != 0))
        {
            // skip
            continue;
        }

#if CONFIG_SETTINGS_DEBUG
        formatln("config file: section '%s' start", section_descriptor->vtbl->name);
#endif
        section_descriptor->vtbl->init(section_descriptor);

        input_stream_t ins;

        if(ISOK(err = file_input_stream_open(&ins, configuration_file_path)))
        {
            // parse stream will parse ALL sections

            if(ISOK(err = config_file_reader_parse_stream(configuration_file_path, &ins, section_descriptor, cfgerr)))
            {
                // whatever
#if CONFIG_SETTINGS_DEBUG
                formatln("config file: section '%s' done", section_descriptor->vtbl->name);
#endif
                if((config_autodefault_after_source != 0) && (config_autodefault_after_source == config_current_source))
                {
#if CONFIG_SETTINGS_DEBUG
                    formatln("config file: section '%s' applying default", section_descriptor->vtbl->name);
#endif
                    // apply default

                    config_current_source = config_default_source;

                    err = config_set_section_default(section_descriptor, cfgerr);

                    config_current_source = config_autodefault_after_source;

                    if(FAIL(err))
                    {
                        break;
                    }
                }

                if((err = config_fire_on_section_read_callback(section_descriptor->vtbl->name, -1)) != 0)
                {
                    break;
                }
            }
            else
            {
                break;
            }
        }
        else
        {
#if DEBUG
            formatln("config file: cannot open: '%s': %r", configuration_file_path, err);
#endif
            break;
        }

        if(section_descriptor->vtbl->postprocess != NULL)
        {
            if(FAIL(err = section_descriptor->vtbl->postprocess(section_descriptor, cfgerr)))
            {
                if(cfgerr != NULL && !cfgerr->has_content)
                {
                    cfgerr->line_number = 0;
                    cfgerr->has_content = true;
                    cfgerr->line[0] = '\0';
                    snformat(cfgerr->file, sizeof(cfgerr->file), "section %s", section_descriptor->vtbl->name);
                }

                break;
            }
        }
    }

    return err;
}

ya_result config_read(const char *configuration_file_path, config_error_t *cfgerr)
{
    ya_result return_code = config_read_section(configuration_file_path, cfgerr, NULL);

    return return_code;
}

/**
 *
 * @param buffer      source buffer
 * @param buffer_len  source buffer size
 * @param buffer_name name of the buffer for error reporting ie: "command-line"
 * @param cfgerr      error handling structure
 *
 * @return an error code
 */

ya_result config_read_from_buffer(const char *buffer, uint32_t buffer_len, const char *buffer_name, config_error_t *cfgerr)
{
    ya_result              err = SUCCESS;

    u32_treemap_iterator_t iter;
    u32_treemap_iterator_init(&section_descriptor_set, &iter);
    while(u32_treemap_iterator_hasnext(&iter))
    {
        u32_treemap_node_t          *node = u32_treemap_iterator_next_node(&iter);
        config_section_descriptor_t *section_descriptor = (config_section_descriptor_t *)node->value;

#if CONFIG_SETTINGS_DEBUG
        formatln("config buffer: section '%s' start", section_descriptor->vtbl->name);
#endif
        section_descriptor->vtbl->init(section_descriptor);

        input_stream_t ins;

        bytearray_input_stream_init_const(&ins, (const uint8_t *)buffer, buffer_len);

        err = config_file_reader_parse_stream(buffer_name, &ins, section_descriptor, cfgerr);

        /// @note config_file_reader_parse_stream closes the stream
        // DO NOT: input_stream_close(&ins);

        if(ISOK(err))
        {
            // whatever

#if CONFIG_SETTINGS_DEBUG
            formatln("config buffer: section '%s' done", section_descriptor->vtbl->name);
#endif
        }
        else
        {
            break;
        }
    }

    return err;
}

static ya_result config_source_get_from_buffer(struct config_source_s *source, input_stream_t *out_stream, config_error_t *cfgerr)
{
    (void)cfgerr;
    bytearray_input_stream_init_const(out_stream, (const uint8_t *)source->source.buffer.text, source->source.buffer.size);
    return SUCCESS;
}

void config_source_set_buffer(struct config_source_s *source, const char *name, uint8_t level, const char *buffer, uint32_t buffer_len)
{
    source->get_source = config_source_get_from_buffer;
    source->name = name;
    source->__class__ = "buffer_source";
    source->source.buffer.text = buffer;
    source->source.buffer.size = buffer_len;
    source->level = level;
}

static ya_result config_source_get_from_file(struct config_source_s *source, input_stream_t *out_stream, config_error_t *cfgerr)
{
    ya_result return_code;
    if(source->source.file_name.name != NULL)
    {
        return_code = file_input_stream_open(out_stream, source->source.file_name.name);
        if(FAIL(return_code))
        {
            if(cfgerr != NULL)
            {
                strcpy_ex(cfgerr->file, source->source.file_name.name, sizeof(cfgerr->file));
                strcpy_ex(cfgerr->line, "unable to open file", sizeof(cfgerr->line));
                cfgerr->line_number = 0;
            }
        }
    }
    else
    {
        return CONFIG_INTERNAL_ERROR;
    }
    return return_code;
}

/**
 * Sets-up a file source
 *
 * @param source the source struct to initialise
 * @param name the name of the file
 * @param level the level of the source
 */

void config_source_set_file(struct config_source_s *source, const char *name, uint8_t level)
{
    source->get_source = config_source_get_from_file;
    source->name = name;
    source->__class__ = "file_source";
    source->source.file_name.name = name;
    source->level = level;
}

/**
 * Sets-up a command-line source.
 *
 * @param source the source
 * @param cmdline a command line descriptor
 * @param argc ... argc
 * @param argv ... argv
 *
 * @return an error code
 */

ya_result config_source_set_commandline(struct config_source_s *source, const cmdline_desc_t *cmdline, int argc, char **argv)
{
    input_stream_t config_is;
    ya_result      ret;

    source->get_source = config_source_get_from_buffer;
    source->name = "command line";
    source->__class__ = "command line source";
    source->source.file_name.name = "command line";
    source->level = CONFIG_SOURCE_CMDLINE;

    int argc_error;

    if(FAIL(ret = cmdline_parse(cmdline, argc, argv, NULL, NULL, &config_is, &argc_error)))
    {
        if(argc_error > 0)
        {
            formatln("command line: %r at '%s'", ret, argv[argc_error]);
        }
        else
        {
            formatln("command line: %r", ret);
        }

        flushout();

        return ret;
    }

    source->source.buffer.size = bytearray_input_stream_size(&config_is);
    source->source.buffer.text = (char *)bytearray_input_stream_detach(&config_is);

    input_stream_close(&config_is);

    return SUCCESS;
}

/**
 * Read all sources from a table
 *
 * @param sources a pointer to the first source
 * @param sources_count the number of sources
 * @param cfgerr if not NULL, the error reporting structure
 *
 * @return an error code
 *
 * Example:
 *
 * config_error_t cerr;
 * config_error_init(&cerr);
 * struct config_source_s sources[3];                                       // needs to be sorted by decreasing source
 * level config_source_set_commandline(&sources[0], config_cmdline, argc, argv);  // level = CONFIG_SOURCE_CMDLINE = 250
 * config_source_set_buffer(&sources[1], "local", 3, config_conf_buffer, config_conf_buffer_size);
 * config_source_set_file(&sources[2], CONFIG_FILE_NAME, 2);
 * ret = config_read_from_sources(sources, 3, &cerr);
 */

ya_result config_read_from_sources(struct config_source_s *sources, uint32_t sources_count, config_error_t *cfgerr)
{
    ya_result err = SUCCESS;

    config_error_reset(cfgerr);

    // test that the sources are properly set

    uint8_t last_source = U8_MAX;
    for(uint_fast32_t i = 0; i < sources_count; i++)
    {
        if(sources[i].get_source == NULL)
        {
            return CONFIG_INTERNAL_ERROR; // NULL callback
        }

        if(sources[i].level > last_source)
        {
            return CONFIG_INTERNAL_ERROR; // sources are not sorted
        }

        last_source = sources[i].level;
    }

    if(last_source <= 1)
    {
        return CONFIG_INTERNAL_ERROR; // do not put "default" nor "none" in a source level
    }

    u32_treemap_iterator_t iter;
    u32_treemap_iterator_init(&section_descriptor_set, &iter);
    while(u32_treemap_iterator_hasnext(&iter))
    {
        u32_treemap_node_t          *node = u32_treemap_iterator_next_node(&iter);
        config_section_descriptor_t *section_descriptor = (config_section_descriptor_t *)node->value;

#if CONFIG_SETTINGS_DEBUG
        formatln("config buffer: section '%s' start", section_descriptor->vtbl->name);
#endif

        section_descriptor->vtbl->init(section_descriptor);

        // command line

        input_stream_t ins;

        for(uint_fast32_t source_index = 0; source_index < sources_count; source_index++)
        {
            struct config_source_s *source = &sources[source_index];
#if CONFIG_SETTINGS_DEBUG
            formatln("config buffer: section '%s' getting source '%s'", section_descriptor->vtbl->name, source->name);
#endif
            config_set_source(source->level);

            // retrieve the stream

            if(FAIL(err = sources[source_index].get_source(source, &ins, cfgerr)))
            {
                break;
            }

#if CONFIG_SETTINGS_DEBUG
            formatln("config buffer: section '%s' parsing stream", section_descriptor->vtbl->name);
#endif
            if(FAIL(err = config_file_reader_parse_stream(source->name, &ins, section_descriptor, cfgerr)))
            {
                break;
            }

#if CONFIG_SETTINGS_DEBUG
            formatln("config buffer: section '%s' parsing stream done", section_descriptor->vtbl->name);
#endif
            // note: ins must be closed
        }

        // default

        if(ISOK(err))
        {
            config_set_source(CONFIG_SOURCE_DEFAULT);

            if(FAIL(err = config_set_section_default(section_descriptor, cfgerr)))
            {
#if !CONFIG_SECTION_DESCRIPTOR_TRACK
                if((cfgerr != NULL) && !cfgerr->has_content)
                {
                    cfgerr->variable_name = "";
                    cfgerr->line_number = 0;
                    cfgerr->has_content = true;
                    cfgerr->line[0] = '\0';
                    snformat(cfgerr->file, sizeof(cfgerr->file), "setting-up section %s default", section_descriptor->vtbl->name);
                }
#endif
                break;
            }

            if(section_descriptor->vtbl->postprocess != NULL)
            {
                if(FAIL(err = section_descriptor->vtbl->postprocess(section_descriptor, cfgerr)))
                {
#if !CONFIG_SECTION_DESCRIPTOR_TRACK
                    if((cfgerr != NULL) && !cfgerr->has_content)
                    {
                        cfgerr->variable_name = "";
                        cfgerr->line_number = 0;
                        cfgerr->has_content = true;
                        cfgerr->line[0] = '\0';
                        snformat(cfgerr->file, sizeof(cfgerr->file), "section %s", section_descriptor->vtbl->name);
                    }
#endif
                    break;
                }
            }

            // callback

            if((err = config_fire_on_section_read_callback(section_descriptor->vtbl->name, -1)) != 0)
            {
                if(FAIL(err))
                {
                    if((cfgerr != NULL) && !cfgerr->has_content)
                    {
                        config_error_set_variable_name(cfgerr, NULL, false);
                        cfgerr->line_number = 0;
                        cfgerr->has_content = false;
                        cfgerr->line[0] = '\0';
                        cfgerr->file[0] = '\0';
                    }
                }
                break;
            }
        }
        else
        {
            if(err != MAKE_ERRNO_ERROR(ENOENT))
            {
                break;
            }
        }
    }

    config_set_source(CONFIG_SOURCE_NONE);

    return err;
}

ya_result config_set_default(config_error_t *cfgerr)
{
    ya_result              err = SUCCESS;

    u32_treemap_iterator_t iter;
    u32_treemap_iterator_init(&section_descriptor_set, &iter);
    while(u32_treemap_iterator_hasnext(&iter))
    {
        u32_treemap_node_t          *node = u32_treemap_iterator_next_node(&iter);
        config_section_descriptor_t *section_descriptor = (config_section_descriptor_t *)node->value;

#if CONFIG_SETTINGS_DEBUG
        formatln("config default: section '%s' start", section_descriptor->vtbl->name);
#endif

        err = config_set_section_default(section_descriptor, cfgerr);

#if CONFIG_SETTINGS_DEBUG
        formatln("config default: section '%s' done", section_descriptor->vtbl->name);
#endif

        if(ISOK(err))
        {
            // whatever

            log_debug("config default: section '%s'", section_descriptor->vtbl->name);
        }
        else
        {
            break;
        }
    }

    return err;
}

ya_result config_value_set_to_default(const char *section_name, const char *name, config_error_t *cfgerr)
{
    config_section_descriptor_t *section_descriptor = NULL;
    ya_result                    err = CONFIG_SECTION_ERROR;

    u32_treemap_iterator_t       iter;
    u32_treemap_iterator_init(&section_descriptor_set, &iter);
    while(u32_treemap_iterator_hasnext(&iter))
    {
        u32_treemap_node_t          *node = u32_treemap_iterator_next_node(&iter);
        config_section_descriptor_t *section_desc = (config_section_descriptor_t *)node->value;

        if(strcmp(section_desc->vtbl->name, section_name) == 0)
        {
            section_descriptor = section_desc;
            break;
        }
    }

    if(section_descriptor != NULL)
    {
        if(section_descriptor->vtbl->table != NULL)
        {
            section_descriptor->vtbl->init(section_descriptor);

            if(ISOK(err = config_item_index_get(section_descriptor->vtbl->table, name)))
            {
                const config_table_descriptor_item_t *item = &section_descriptor->vtbl->table[err];

                if(item->default_value_string != NULL)
                {
                    if(FAIL(err = config_value_set(section_descriptor, item->name, item->default_value_string, cfgerr)))
                    {
                        if((cfgerr != NULL) && !cfgerr->has_content)
                        {
                            config_error_set_variable_name(cfgerr, NULL, false);
                            cfgerr->line_number = 0;
                            cfgerr->has_content = true;
                            strcpy_ex(cfgerr->line, item->name, sizeof(cfgerr->line));
                            strcpy_ex(cfgerr->file, STRNULL(item->default_value_string), sizeof(cfgerr->file));
                        }
                    }
                }
            }
        }
    }

    return err;
}

ya_result config_value_get_source(const char *section_name, const char *name)
{
    config_section_descriptor_t *section_descriptor = NULL;
    ya_result                    ret = CONFIG_SECTION_ERROR;

    u32_treemap_iterator_t       iter;
    u32_treemap_iterator_init(&section_descriptor_set, &iter);
    while(u32_treemap_iterator_hasnext(&iter))
    {
        u32_treemap_node_t          *node = u32_treemap_iterator_next_node(&iter);
        config_section_descriptor_t *section_desc = (config_section_descriptor_t *)node->value;

        if(strcmp(section_desc->vtbl->name, section_name) == 0)
        {
            section_descriptor = section_desc;
            break;
        }
    }

    if(section_descriptor != NULL)
    {
        if(section_descriptor->vtbl->table != NULL)
        {
            section_descriptor->vtbl->init(section_descriptor);

            if(ISOK(ret = config_item_index_get(section_descriptor->vtbl->table, name)))
            {
                const config_table_descriptor_item_t *item = &section_descriptor->vtbl->table[ret];

                ret = item->source;
            }
        }
    }

    return ret;
}

static inline void config_item_name_canonize(const char *name, char *filtered_name)
{
    size_t name_len = strlen(name);

    for(size_t i = 0; i < name_len; i++)
    {
        char c = name[i];
        if((c == '-') || (c == '.'))
        {
            c = '_';
        }
        filtered_name[i] = c;
    }
    filtered_name[name_len] = '\0';
}

bool config_section_struct_register_type_handler(config_set_field_function *setter, config_section_struct_type_handler *handler)
{
    // workaround
    config_set_field_function_as_voidp key;
    key.setter = setter;

    ptr_treemap_node_t *node = ptr_treemap_insert(&config_section_struct_type_handler_set, key.ptr);
    if(node->value == NULL)
    {
        config_section_struct_type_handler_as_voidp value;
        value.handler = handler;

        node->value = value.ptr;
        return true;
    }
    else
    {
        return false;
    }
}

void config_section_struct_print(const config_section_descriptor_t *section_descriptor, const void *configbase, output_stream_t *os)
{
    const char                           *value;
    const config_table_descriptor_item_t *table = section_descriptor->vtbl->table;

    char                                  tmp[1024];

    if(configbase == NULL)
    {
        return;
    }

    while(table->name != NULL)
    {
        bool already = false;

        char filtered_name[128];

        config_item_name_canonize(table->name, filtered_name);

        /* table->setter is NULL for aliases */
        if(table->setter != NULL)
        {
            intptr_t base = (intptr_t)configbase;
            intptr_t offs = (intptr_t)table->field_offset;
            void    *ptr = (void *)(base + offs);

            if(table->setter == (config_set_field_function *)config_set_bool)
            {
                bool b = *(bool *)ptr;
                value = (b) ? "yes" : "no";
            }
            else if(table->setter == (config_set_field_function *)config_set_flag8)
            {
                uint8_t *f = (uint8_t *)ptr;
                bool     b = *f & table->function_specific._u8;
                value = (b) ? "yes" : "no";
            }
            else if(table->setter == (config_set_field_function *)config_set_flag16)
            {
                uint16_t *f = (uint16_t *)ptr;
                bool      b = *f & table->function_specific._u16;
                value = (b) ? "yes" : "no";
            }
            else if(table->setter == (config_set_field_function *)config_set_flag32)
            {
                uint32_t *f = (uint32_t *)ptr;
                bool      b = *f & table->function_specific._u32;
                value = (b) ? "yes" : "no";
            }
            else if(table->setter == (config_set_field_function *)config_set_flag64)
            {
                uint64_t *f = (uint64_t *)ptr;
                bool      b = *f & table->function_specific._u64;
                value = (b) ? "yes" : "no";
            }
            else if(table->setter == (config_set_field_function *)config_set_u64)
            {
                uint64_t *v = (uint64_t *)ptr;
                snformat(tmp, sizeof(tmp), "%lld", *v);
                value = tmp;
            }
            else if((table->setter == (config_set_field_function *)config_set_u32) || (table->setter == (config_set_field_function *)config_set_u32_range) || (table->setter == (config_set_field_function *)config_set_u32_clamp))
            {
                uint32_t *v = (uint32_t *)ptr;
                snformat(tmp, sizeof(tmp), "%d", *v);
                value = tmp;
            }
            else if(table->setter == (config_set_field_function *)config_set_s32)
            {
                int32_t *v = (int32_t *)ptr;
                snformat(tmp, sizeof(tmp), "%i", *v);
                value = tmp;
            }
            else if(table->setter == (config_set_field_function *)config_set_u16)
            {
                uint16_t *v = (uint16_t *)ptr;
                snformat(tmp, sizeof(tmp), "%d", *v);
                value = tmp;
            }
            else if((table->setter == (config_set_field_function *)config_set_u8) || (table->setter == (config_set_field_function *)config_inc_u8))
            {
                uint8_t *v = (uint8_t *)ptr;
                snformat(tmp, sizeof(tmp), "%d", *v);
                value = tmp;
            }
            else if(table->setter == (config_set_field_function *)config_set_uid_t)
            {
                uid_t *v = (uid_t *)ptr;
                snformat(tmp, sizeof(tmp), "%d", *v);
                value = tmp;
            }
            else if(table->setter == (config_set_field_function *)config_set_gid_t)
            {
                gid_t *v = (gid_t *)ptr;
                snformat(tmp, sizeof(tmp), "%d", *v);
                value = tmp;
            }
            else if((table->setter == (config_set_field_function *)config_set_string) || (table->setter == (config_set_field_function *)config_set_path) || (table->setter == (config_set_field_function *)config_set_logpath) ||
                    (table->setter == (config_set_field_function *)config_set_file))
            {
                value = *((char **)ptr);
                if((value == NULL) || (strlen(value) == 0))
                {
                    value = "\"\"";
                }
            }
            else if(table->setter == (config_set_field_function *)config_set_password)
            {
                value = "????????";
            }
#if __unix__
            else if(table->setter == (config_set_field_function *)config_set_chroot)
            {
                value = chroot_get_path();
            }
#endif
            else if(table->setter == (config_set_field_function *)config_set_dnstype)
            {
                uint16_t *v = (uint16_t *)ptr;
                value = dns_type_get_name(*v);
            }
            else if(table->setter == (config_set_field_function *)config_set_dnsclass)
            {
                uint16_t *v = (uint16_t *)ptr;
                value = dns_class_get_name(*v);
            }
            else if(table->setter == (config_set_field_function *)config_set_string_copy)
            {
                value = (char *)ptr;
                if((value == NULL) || (strlen(value) == 0))
                {
                    value = "\"\"";
                }
            }
            else if(table->setter == (config_set_field_function *)config_set_fqdn)
            {
                snformat(tmp, sizeof(tmp), "%{dnsname}", *((uint8_t **)ptr));
                value = tmp;
            }
            else if(table->setter == (config_set_field_function *)acl_config_set_item)
            {
                address_match_set_t *ams = (address_match_set_t *)ptr;
                if(ams != NULL)
                {
                    osformat(os, "%24s", filtered_name);
                    acl_address_match_set_to_stream(os, ams);
                    osprintln(os, "");
                }
                else
                {
                    osformatln(os, "# '%s' is empty", filtered_name);
                }
                already = true;
                value = NULL;
            }
            else if(table->setter == (config_set_field_function *)config_set_host_list)
            {
                host_address_t *v = *(host_address_t **)ptr;

                if(v != NULL)
                {
                    osformat(os, "%24s", filtered_name);

                    char sep = ' ';

                    do
                    {
                        socketaddress_t sa;
                        host_address2sockaddr(v, &sa);
                        osformat(os, "%c%{sockaddrip}", sep, &sa);
                        if(v->port != NU16(DNS_DEFAULT_PORT))
                        {
                            osformat(os, " port %hd", ntohs(v->port));
                        }
#if DNSCORE_HAS_TSIG_SUPPORT
                        if(v->tsig != NULL)
                        {
                            osformat(os, " key %{dnsname}", v->tsig->name);
                        }
#endif
                        sep = ',';

                        v = v->next;
                    } while(v != NULL);

                    osprintln(os, "");
                }
                else
                {
                    osformatln(os, "# '%s' is empty", filtered_name);
                }

                already = true;
                value = NULL;
            }
            else if(table->setter == (config_set_field_function *)config_set_enum_value)
            {
                uint32_t           *v = (uint32_t *)ptr;

                value_name_table_t *tbl = table->function_specific._voidp;

                value = "?";

                while(tbl->data != NULL)
                {
                    if(tbl->id == *v)
                    {
                        value = tbl->data;
                        break;
                    }

                    tbl++;
                }
            }
            else if(table->setter == (config_set_field_function *)config_set_enum8_value)
            {
                uint8_t            *v = (uint8_t *)ptr;

                value_name_table_t *tbl = table->function_specific._voidp;

                value = "?";

                while(tbl->data != NULL)
                {
                    if(tbl->id == *v)
                    {
                        value = tbl->data;
                        break;
                    }

                    tbl++;
                }
            }
            else if(table->setter == (config_set_field_function *)config_set_bytes)
            {
                uint8_t *v = (uint8_t *)ptr;

                if(v != NULL)
                {
                    uint32_t v_size = table->function_specific._u32;

                    assert(v_size < (sizeof(tmp) * 3 + 3) / 4);

                    uint32_t e_size = base64_encode(v, v_size, tmp);
                    tmp[e_size] = '\0';

                    value = tmp;
                }
                else
                {
                    value = NULL;
                }
            }
            /*
            else if(table->setter == (config_set_field_function*)acl_config_set_access_control_item)
            {
            }
            */
            else
            {
                config_set_field_function_as_voidp key;
                key.setter = table->setter;

                ptr_treemap_node_t *node = ptr_treemap_find(&config_section_struct_type_handler_set, key.ptr);

                if(node != NULL)
                {
                    config_section_struct_type_handler_as_voidp alias_value;
                    alias_value.ptr = node->value;
                    config_section_struct_type_handler *type_handler = alias_value.handler;
                    already = type_handler(os, table->name, ptr);
                    value = NULL;
                }
                else
                {
                    osformatln(os, "# '%s' parameter cannot be dumped", filtered_name);
                    value = NULL;
                    already = true;
                }
            }

            if(!already)
            {
                if(value != NULL)
                {
                    osformatln(os, "%s %s", filtered_name, value);
                }
#if DEBUG
                else
                {
                    osformatln(os, "# '%s' is not set", filtered_name);
                }
#endif
            }
        }
        table++;
    }
}

/**
 * Frees the content of a struct using the fields described in its associated config_section_descriptor_t
 *
 * @param section_descriptor the descriptor
 * @param configbase the structure holding the configuration
 */

void config_section_struct_free(const config_section_descriptor_t *section_descriptor, void *configbase)
{
    const config_table_descriptor_item_t *table = section_descriptor->vtbl->table;

    if(configbase == NULL)
    {
        return;
    }

    while(table->name != NULL)
    {
        char filtered_name[128];

        config_item_name_canonize(table->name, filtered_name);

        /* table->setter is NULL for aliases */
        if(table->setter != NULL)
        {
            intptr_t base = (intptr_t)configbase;
            intptr_t offs = (intptr_t)table->field_offset;
            void    *ptr = (void *)(base + offs);

            if(table->setter == (config_set_field_function *)config_set_bool)
            {
            }
            else if(table->setter == (config_set_field_function *)config_set_flag8)
            {
            }
            else if(table->setter == (config_set_field_function *)config_set_flag16)
            {
            }
            else if(table->setter == (config_set_field_function *)config_set_flag32)
            {
            }
            else if(table->setter == (config_set_field_function *)config_set_flag64)
            {
            }
            else if(table->setter == (config_set_field_function *)config_set_u64)
            {
            }
            else if((table->setter == (config_set_field_function *)config_set_u32) || (table->setter == (config_set_field_function *)config_set_u32_range) || (table->setter == (config_set_field_function *)config_set_u32_clamp))
            {
            }
            else if(table->setter == (config_set_field_function *)config_set_s32)
            {
            }
            else if(table->setter == (config_set_field_function *)config_set_u16)
            {
            }
            else if((table->setter == (config_set_field_function *)config_set_u8) || (table->setter == (config_set_field_function *)config_inc_u8))
            {
            }
            else if(table->setter == (config_set_field_function *)config_set_uid_t)
            {
            }
            else if(table->setter == (config_set_field_function *)config_set_gid_t)
            {
            }
            else if(table->setter == (config_set_field_function *)config_append_string_array_item)
            {
                ptr_vector_t *array = (ptr_vector_t *)ptr;
                if(array->data != NULL)
                {
                    for(int i = 0; i <= ptr_vector_last_index(array); ++i)
                    {
                        char *text = (char *)ptr_vector_get(array, i);
                        if((text != NULL))
                        {
                            free(text);
                        }
                    }
                }
                ptr_vector_finalise(array);
            }
            else if(table->setter == (config_set_field_function *)config_set_password)
            {
                char *text = *((char **)ptr);
                if((text != NULL))
                {
                    size_t text_len = strlen(text);
                    memset(text, 0, text_len);
                    free(text);
                    *((char **)ptr) = NULL;
                }
            }
            else if((table->setter == (config_set_field_function *)config_set_string) || (table->setter == (config_set_field_function *)config_set_path) || (table->setter == (config_set_field_function *)config_set_logpath) ||
                    (table->setter == (config_set_field_function *)config_set_chroot) || (table->setter == (config_set_field_function *)config_set_file))
            {
                char *text = *((char **)ptr);
                if((text != NULL))
                {
                    free(text);
                    *((char **)ptr) = NULL;
                }
            }
            else if(table->setter == (config_set_field_function *)config_set_password)
            {
            }
#if __unix__
            else if(table->setter == (config_set_field_function *)config_set_chroot)
            {
                chroot_set_path(NULL);
            }
#endif
            else if(table->setter == (config_set_field_function *)config_set_dnstype)
            {
            }
            else if(table->setter == (config_set_field_function *)config_set_dnsclass)
            {
            }
            else if(table->setter == (config_set_field_function *)config_set_string_copy)
            {
            }
            else if(table->setter == (config_set_field_function *)config_set_fqdn)
            {
                uint8_t *fqdn = *((uint8_t **)ptr);
                if(fqdn != NULL)
                {
                    free(fqdn);
                    *((uint8_t **)ptr) = NULL;
                }
            }
            else if(table->setter == (config_set_field_function *)config_set_host_list)
            {
                host_address_t *v = *(host_address_t **)ptr;

                if(v != NULL)
                {
                    host_address_delete_list(v);
                    *(host_address_t **)ptr = NULL;
                }
            }
            else if(table->setter == (config_set_field_function *)config_set_enum_value)
            {
            }
            else if(table->setter == (config_set_field_function *)config_set_enum8_value)
            {
            }
            else if(table->setter == (config_set_field_function *)config_set_bytes)
            {
            }
            else
            {
                config_set_field_function_as_voidp key;
                key.setter = table->setter;

                ptr_treemap_node_t *node = ptr_treemap_find(&config_section_struct_type_handler_set, key.ptr);

                if(node != NULL)
                {
                    // config_section_struct_type_handler_as_voidp alias_value;
                    // alias_value.ptr = node->value;
                    // config_section_struct_type_handler *type_handler = alias_value.handler;
                    // value = NULL;
                }
                else
                {
                }
            }
        }
        table++;
    }
}

void config_section_print(const config_section_descriptor_t *section_descriptor, output_stream_t *os) { config_section_struct_print(section_descriptor, section_descriptor->base, os); }

void config_print(output_stream_t *os)
{
    u32_treemap_iterator_t iter;
    u32_treemap_iterator_init(&section_descriptor_set, &iter);
    while(u32_treemap_iterator_hasnext(&iter))
    {
        u32_treemap_node_t          *node = u32_treemap_iterator_next_node(&iter);
        config_section_descriptor_t *section_descriptor = (config_section_descriptor_t *)node->value;

        if((section_descriptor->vtbl->table != NULL) && (section_descriptor->base != NULL))
        {
            osformatln(os, "<%s>", section_descriptor->vtbl->name);
            config_section_print(section_descriptor, os);
            osformatln(os, "</%s>\n", section_descriptor->vtbl->name);
        }
        else // there is not fixed structure associated to the configuration
        {
            // note: never stop iterating before the updated context value is NULL

            void *context = NULL;

            do
            {
                osformatln(os, "<%s>", section_descriptor->vtbl->name);
                section_descriptor->vtbl->print_wild(section_descriptor, os, NULL, &context);
                osformatln(os, "</%s>\n", section_descriptor->vtbl->name);
            } while(context != NULL);
        }
    }
}

ya_result config_item_index_get(const config_table_descriptor_item_t *table, const char *name)
{
    if(table != NULL)
    {
        // replaces all '-' and '.' by '_'

        char filtered_name[128];

        config_item_name_canonize(name, filtered_name);

        int count = 0;

        while(table[count].name != NULL)
        {
            if(strchr(table[count].name, '.') == NULL)
            {
                if(strcasecmp(table[count].name, filtered_name) == 0)
                {
                    return count;
                }
            }
            else
            {
                char table_filtered_name[128];
                config_item_name_canonize(table[count].name, table_filtered_name);
                if(strcasecmp(table_filtered_name, filtered_name) == 0)
                {
                    return count;
                }
            }

            count++;
        }
    }

    return CONFIG_UNKNOWN_SETTING; /* not found */
}

config_section_descriptor_t *config_section_get_descriptor(const char *name)
{
    u32_treemap_iterator_t iter;
    u32_treemap_iterator_init(&section_descriptor_set, &iter);
    while(u32_treemap_iterator_hasnext(&iter))
    {
        u32_treemap_node_t          *node = u32_treemap_iterator_next_node(&iter);
        config_section_descriptor_t *section_descriptor = (config_section_descriptor_t *)node->value;

        if(strcmp(section_descriptor->vtbl->name, name) == 0)
        {
            return section_descriptor;
        }
    }

    return NULL;
}

ya_result config_value_set(config_section_descriptor_t *section_descriptor, const char *key, const char *value, config_error_t *cfgerr)
{
    (void)cfgerr;
    ya_result err;
    int8_t    maxalias = 16;

    // seek the entry, going through aliases if needs to be

    for(;;)
    {
        if(FAIL(err = config_item_index_get(section_descriptor->vtbl->table, key)))
        {
            if(section_descriptor->vtbl->set_wild != NULL)
            {
                err = section_descriptor->vtbl->set_wild(section_descriptor, key, value);
            }

            return err;
        }

        if(section_descriptor->vtbl->table[err].setter != NULL)
        {
            break;
        }

        if(--maxalias <= 0)
        {
            return CONFIG_ALIAS_CHAIN_TOO_BIG; // alias chain too big
        }

        // point to the aliased

        key = section_descriptor->vtbl->table[err].default_value_string;
    }

    config_table_descriptor_item_t *item = &section_descriptor->vtbl->table[err];

    // only set it if the field has not been set by a source beyond the current one

    if(item->source <= config_current_source)
    {
        uint8_t *base = (uint8_t *)section_descriptor->base;

        log_debug1("config: '%s' setting '%s' to '%s'", STRNULL(section_descriptor->vtbl->name), STRNULL(key), STRNULL(value));

        if(ISOK(err = item->setter(value, &base[item->field_offset], item->function_specific)))
        {
#if CONFIG_SECTION_DESCRIPTOR_TRACK
            config_section_descriptor_file_line_add(section_descriptor, &base[item->field_offset], cfgerr->file, cfgerr->line_number);
#endif
            item->source = config_current_source;
        }
    }
    else
    {
        log_debug1("config: '%s' has already been set by source %u (current is %u)", item->name, item->source, config_current_source);
    }

    return err;
}

#if NOTUSED
ya_result config_source_set_by_target(config_section_descriptor_t *section_descriptor, void *target_ptr)
{
    config_table_descriptor_item_s *item = section_descriptor->vtbl->table;
    uint8_t                        *base = (uint8_t *)section_descriptor->base;

    while(item->name != NULL)
    {
        if((void *)&base[item->field_offset] == target_ptr)
        {
            item->source = config_default_source;
            return SUCCESS;
        }
    }

    return ERROR;
}
#endif

ya_result config_postprocess()
{
    ya_result              return_code = SUCCESS;

    u32_treemap_iterator_t iter;
    u32_treemap_iterator_init(&section_descriptor_set, &iter);
    while(u32_treemap_iterator_hasnext(&iter))
    {
        u32_treemap_node_t          *node = u32_treemap_iterator_next_node(&iter);
        config_section_descriptor_t *section_descriptor = (config_section_descriptor_t *)node->value;

        if(section_descriptor->vtbl->postprocess != NULL)
        {
            if(FAIL(return_code = section_descriptor->vtbl->postprocess(section_descriptor, NULL)))
            {
                return return_code;
            }
        }
    }

    return SUCCESS;
}

ya_result config_finalize()
{
    ya_result              return_code = SUCCESS;

    u32_treemap_iterator_t iter;
    u32_treemap_iterator_init(&section_descriptor_set, &iter);
    while(u32_treemap_iterator_hasnext(&iter))
    {
        u32_treemap_node_t          *node = u32_treemap_iterator_next_node(&iter);
        config_section_descriptor_t *section_descriptor = (config_section_descriptor_t *)node->value;

        if(section_descriptor->vtbl->finalise != NULL)
        {
            if(FAIL(return_code = section_descriptor->vtbl->finalise(section_descriptor)))
            {
                return return_code;
            }
        }
    }

    u32_treemap_finalise(&section_descriptor_set);

    ptr_treemap_iterator_t iter2;
    ptr_treemap_iterator_init(&on_section_read_callback_set, &iter2);
    while(ptr_treemap_iterator_hasnext(&iter2))
    {
        ptr_treemap_node_t *node = ptr_treemap_iterator_next_node(&iter2);
        free(node->key);
    }

    ptr_treemap_finalise(&on_section_read_callback_set);

    return SUCCESS;
}

/** @} */
