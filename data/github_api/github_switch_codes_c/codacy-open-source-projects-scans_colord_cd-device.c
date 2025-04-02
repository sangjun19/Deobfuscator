/* -*- Mode: C; tab-width: 8; indent-tabs-mode: t; c-basic-offset: 8 -*-
 *
 * Copyright (C) 2010-2012 Richard Hughes <richard@hughsie.com>
 *
 * Licensed under the GNU General Public License Version 2
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include "config.h"

#include <glib-object.h>
#include <gio/gio.h>
#include <sys/time.h>
#include <string.h>
#ifdef HAVE_PWD_H
#include <pwd.h>
#endif

#include "cd-common.h"
#include "cd-device.h"
#include "cd-mapping-db.h"
#include "cd-device-db.h"
#include "cd-profile-array.h"
#include "cd-profile.h"
#include "cd-inhibit.h"

static void cd_device_finalize			 (GObject *object);
static void cd_device_dbus_emit_property_changed (CdDevice *device,
						  const gchar *property_name,
						  GVariant *property_value);
static void cd_device_dbus_emit_device_changed	 (CdDevice *device);

#define GET_PRIVATE(o) (cd_device_get_instance_private (o))

typedef struct
{
	CdObjectScope			 object_scope;
	CdProfileArray			*profile_array;
	CdMappingDb			*mapping_db;
	CdDeviceDb			*device_db;
	CdInhibit			*inhibit;
	gchar				*id;
	gchar				*model;
	gchar				*serial;
	gchar				*vendor;
	gchar				*colorspace;
	gchar				*format;
	gchar				*mode;
	CdDeviceKind			 kind;
	gchar				*object_path;
	GDBusConnection			*connection;
	GPtrArray			*profiles; /* of CdDeviceProfileItem */
	guint				 registration_id;
	guint				 watcher_id;
	guint64				 created;
	guint64				 modified;
	gboolean			 require_modified_signal;
	gboolean			 is_virtual;
	gboolean			 enabled;
	gboolean			 embedded;
	GHashTable			*metadata;
	guint				 owner;
	gchar				*seat;
} CdDevicePrivate;

enum {
	SIGNAL_INVALIDATE,
	SIGNAL_LAST
};

enum {
	PROP_0,
	PROP_OBJECT_PATH,
	PROP_ID,
	PROP_LAST
};

typedef struct {
	CdProfile		*profile;
	CdDeviceRelation	 relation;
	guint64			 timestamp;
} CdDeviceProfileItem;

static guint signals[SIGNAL_LAST] = { 0 };
G_DEFINE_TYPE_WITH_PRIVATE (CdDevice, cd_device, G_TYPE_OBJECT)

GQuark
cd_device_error_quark (void)
{
	guint i;
	static GQuark quark = 0;
	if (!quark) {
		quark = g_quark_from_static_string ("CdDevice");
		for (i = 0; i < CD_DEVICE_ERROR_LAST; i++) {
			g_dbus_error_register_error (quark,
						     i,
						     cd_device_error_to_string (i));
		}
	}
	return quark;
}

CdObjectScope
cd_device_get_scope (CdDevice *device)
{
	CdDevicePrivate *priv = GET_PRIVATE (device);
	g_return_val_if_fail (CD_IS_DEVICE (device), 0);
	return priv->object_scope;
}

void
cd_device_set_scope (CdDevice *device, CdObjectScope object_scope)
{
	CdDevicePrivate *priv = GET_PRIVATE (device);
	g_return_if_fail (CD_IS_DEVICE (device));
	priv->object_scope = object_scope;
}

guint
cd_device_get_owner (CdDevice *device)
{
	CdDevicePrivate *priv = GET_PRIVATE (device);
	g_return_val_if_fail (CD_IS_DEVICE (device), G_MAXUINT);
	return priv->owner;
}

void
cd_device_set_owner (CdDevice *device, guint owner)
{
	CdDevicePrivate *priv = GET_PRIVATE (device);
	g_return_if_fail (CD_IS_DEVICE (device));
	priv->owner = owner;
}

const gchar *
cd_device_get_seat (CdDevice *device)
{
	CdDevicePrivate *priv = GET_PRIVATE (device);
	g_return_val_if_fail (CD_IS_DEVICE (device), NULL);
	return priv->seat;
}

void
cd_device_set_seat (CdDevice *device, const gchar *seat)
{
	CdDevicePrivate *priv = GET_PRIVATE (device);
	g_return_if_fail (CD_IS_DEVICE (device));
	priv->seat = g_strdup (seat);
}

static const gchar *
_cd_device_mode_to_string (CdDeviceMode device_mode)
{
	if (device_mode == CD_DEVICE_MODE_PHYSICAL)
		return "physical";
	if (device_mode == CD_DEVICE_MODE_VIRTUAL)
		return "virtual";
	return "unknown";
}

static CdDeviceMode
_cd_device_mode_from_string (const gchar *device_mode)
{
	if (g_strcmp0 (device_mode, "physical") == 0)
		return CD_DEVICE_MODE_PHYSICAL;
	if (g_strcmp0 (device_mode, "virtual") == 0)
		return CD_DEVICE_MODE_VIRTUAL;
	return CD_DEVICE_MODE_UNKNOWN;
}

void
cd_device_set_mode (CdDevice *device, CdDeviceMode mode)
{
	CdDevicePrivate *priv = GET_PRIVATE (device);
	g_return_if_fail (CD_IS_DEVICE (device));
	g_free (priv->mode);
	priv->mode = g_strdup (_cd_device_mode_to_string (mode));
}

CdDeviceMode
cd_device_get_mode (CdDevice *device)
{
	CdDevicePrivate *priv = GET_PRIVATE (device);
	g_return_val_if_fail (CD_IS_DEVICE (device), CD_DEVICE_MODE_UNKNOWN);
	return _cd_device_mode_from_string (priv->mode);
}

const gchar *
cd_device_get_object_path (CdDevice *device)
{
	CdDevicePrivate *priv = GET_PRIVATE (device);
	g_return_val_if_fail (CD_IS_DEVICE (device), NULL);
	return priv->object_path;
}

const gchar *
cd_device_get_id (CdDevice *device)
{
	CdDevicePrivate *priv = GET_PRIVATE (device);
	g_return_val_if_fail (CD_IS_DEVICE (device), NULL);
	return priv->id;
}

const gchar *
cd_device_get_model (CdDevice *device)
{
	CdDevicePrivate *priv = GET_PRIVATE (device);
	g_return_val_if_fail (CD_IS_DEVICE (device), NULL);
	return priv->model;
}

CdDeviceKind
cd_device_get_kind (CdDevice *device)
{
	CdDevicePrivate *priv = GET_PRIVATE (device);
	g_return_val_if_fail (CD_IS_DEVICE (device), CD_DEVICE_KIND_UNKNOWN);
	return priv->kind;
}

void
cd_device_set_kind (CdDevice *device, CdDeviceKind kind)
{
	CdDevicePrivate *priv = GET_PRIVATE (device);
	g_return_if_fail (CD_IS_DEVICE (device));
	g_return_if_fail (kind != CD_DEVICE_KIND_UNKNOWN);
	priv->kind = kind;
}

static void
cd_device_set_object_path (CdDevice *device)
{
	CdDevicePrivate *priv = GET_PRIVATE (device);
#ifdef HAVE_PWD_H
	struct passwd *pw;
#endif
	g_autofree gchar *path_owner = NULL;
	g_autofree gchar *path_tmp = NULL;

	/* append the uid to the object path */
#ifdef HAVE_PWD_H
	pw = getpwuid (priv->owner);
	if (priv->owner == 0 ||
	    g_strcmp0 (pw->pw_name, DAEMON_USER) == 0) {
		path_tmp = g_strdup (priv->id);
	} else {
		path_tmp = g_strdup_printf ("%s_%s_%u",
					    priv->id,
					    pw->pw_name,
					    priv->owner);
	}
#else
	if (priv->owner == 0) {
		path_tmp = g_strdup (priv->id);
	} else {
		path_tmp = g_strdup_printf ("%s_%d",
					    priv->id,
					    priv->owner);
	}
#endif

	/* make sure object path is sane */
	path_owner = cd_main_ensure_dbus_path (path_tmp);
	priv->object_path = g_build_filename (COLORD_DBUS_PATH,
						      "devices",
						      path_owner,
						      NULL);
}

void
cd_device_set_id (CdDevice *device, const gchar *id)
{
	CdDevicePrivate *priv = GET_PRIVATE (device);
	g_autofree gchar *enabled_str = NULL;

	g_return_if_fail (CD_IS_DEVICE (device));

	g_free (priv->id);
	priv->id = g_strdup (id);

	/* now calculate this again */
	cd_device_set_object_path (device);

	/* find initial enabled state */
	enabled_str = cd_device_db_get_property (priv->device_db,
						 priv->id,
						 "Enabled",
						 NULL);
	if (g_strcmp0 (enabled_str, "False") == 0) {
		g_debug ("%s disabled by db at load", id);
		priv->enabled = FALSE;
	} else {
		priv->enabled = TRUE;
	}
}

static void
cd_device_reset_modified (CdDevice *device)
{
	CdDevicePrivate *priv = GET_PRIVATE (device);
	g_debug ("CdDevice: set device Modified");
	priv->modified = g_get_real_time ();
	priv->require_modified_signal = TRUE;
}

static void
cd_device_dbus_emit_property_changed (CdDevice *device,
				      const gchar *property_name,
				      GVariant *property_value)
{
	CdDevicePrivate *priv = GET_PRIVATE (device);
	GVariantBuilder builder;
	GVariantBuilder invalidated_builder;

	/* not yet connected */
	if (priv->connection == NULL)
		return;

	/* build the dict */
	g_variant_builder_init (&invalidated_builder, G_VARIANT_TYPE ("as"));
	g_variant_builder_init (&builder, G_VARIANT_TYPE_ARRAY);
	g_variant_builder_add (&builder,
			       "{sv}",
			       property_name,
			       property_value);
	if (priv->require_modified_signal) {
		g_variant_builder_add (&builder,
				       "{sv}",
				       CD_DEVICE_PROPERTY_MODIFIED,
				       g_variant_new_uint64 (priv->modified));
		priv->require_modified_signal = FALSE;
	}
	g_dbus_connection_emit_signal (priv->connection,
				       NULL,
				       priv->object_path,
				       "org.freedesktop.DBus.Properties",
				       "PropertiesChanged",
				       g_variant_new ("(sa{sv}as)",
				       COLORD_DBUS_INTERFACE_DEVICE,
				       &builder,
				       &invalidated_builder),
				       NULL);
	g_variant_builder_clear (&builder);
	g_variant_builder_clear (&invalidated_builder);
}

static void
cd_device_dbus_emit_device_changed (CdDevice *device)
{
	CdDevicePrivate *priv = GET_PRIVATE (device);

	/* not yet connected */
	if (priv->connection == NULL)
		return;

	/* emit signal */
	g_debug ("CdDevice: emit Changed on %s",
		 cd_device_get_object_path (device));
	g_dbus_connection_emit_signal (priv->connection,
				       NULL,
				       cd_device_get_object_path (device),
				       COLORD_DBUS_INTERFACE_DEVICE,
				       "Changed",
				       NULL,
				       NULL);

	/* emit signal */
	g_debug ("CdDevice: emit Changed");
	g_dbus_connection_emit_signal (priv->connection,
				       NULL,
				       COLORD_DBUS_PATH,
				       COLORD_DBUS_INTERFACE,
				       "DeviceChanged",
				       g_variant_new ("(o)",
							    cd_device_get_object_path (device)),
				       NULL);
}

static gboolean
cd_device_match_qualifier (const gchar *qual1, const gchar *qual2)
{
	guint i;
	g_auto(GStrv) split1 = NULL;
	g_auto(GStrv) split2 = NULL;

	/* split into substring */
	split1 = g_strsplit (qual1, ".", 3);
	split2 = g_strsplit (qual2, ".", 3);

	/* ensure all substrings match */
	for (i = 0; i < 3; i++) {

		/* wildcard in query */
		if (g_strcmp0 (split1[i], "*") == 0)
			continue;

		/* wildcard in qualifier */
		if (g_strcmp0 (split2[i], "*") == 0)
			continue;

		/* exact match */
		if (g_strcmp0 (split1[i], split2[i]) == 0)
			continue;

		/* failed to match substring */
		return FALSE;
	}

	/* success */
	return TRUE;
}

static CdProfile *
cd_device_find_by_qualifier (const gchar *regex,
			     GPtrArray *array,
			     CdDeviceRelation relation)
{
	CdDeviceProfileItem *item;
	const gchar *qualifier;
	gboolean ret;
	guint i;

	/* find using a wildcard */
	for (i = 0; i < array->len; i++) {
		item = g_ptr_array_index (array, i);
		if (item->relation != relation)
			continue;

		/* '*' matches anything, including a blank qualifier */
		if (g_strcmp0 (regex, "*") == 0) {
			g_debug ("anything matches, returning %s",
				 cd_profile_get_id (item->profile));
			return item->profile;
		}

		/* match with a regex */
		qualifier = cd_profile_get_qualifier (item->profile);
		if (qualifier == NULL) {
			g_debug ("no qualifier for %s, skipping",
				 cd_profile_get_id (item->profile));
			continue;
		}
		ret = cd_device_match_qualifier (regex,
						 qualifier);
		g_debug ("%s regex '%s' for '%s'",
			 ret ? "matched" : "unmatched",
			 regex,
			 qualifier);
		if (ret)
			return item->profile;
	}
	return  NULL;
}

static CdProfile *
cd_device_find_profile_by_object_path (GPtrArray *array, const gchar *object_path)
{
	CdDeviceProfileItem *item;
	gboolean ret;
	guint i;

	/* find using an object path */
	for (i = 0; i < array->len; i++) {
		item = g_ptr_array_index (array, i);
		ret = (g_strcmp0 (object_path,
				  cd_profile_get_object_path (item->profile)) == 0);
		if (ret)
			return item->profile;
	}
	return NULL;
}

static GVariant *
cd_device_get_profiles_as_variant (CdDevice *device)
{
	CdDevicePrivate *priv = GET_PRIVATE (device);
	CdDeviceProfileItem *item;
	const gchar *tmp;
	guint i;
	guint idx = 0;
	g_autofree GVariant **profiles = NULL;

	/* Object paths are assembled in this order:
	 *
	 *  1. Hard mapped profiles from the database
	 *  2. Soft mapped profiles of DATA_source != EDID
	 *  2. Soft mapped profiles of DATA_source == EDID
	 */
	profiles = g_new0 (GVariant *, priv->profiles->len + 1);
	for (i = 0; i < priv->profiles->len; i++) {
		item = g_ptr_array_index (priv->profiles, i);
		if (item->relation != CD_DEVICE_RELATION_HARD)
			continue;
		tmp = cd_profile_get_object_path (item->profile);
		profiles[idx++] = g_variant_new_object_path (tmp);
	}
	for (i = 0; i < priv->profiles->len; i++) {
		item = g_ptr_array_index (priv->profiles, i);
		if (item->relation != CD_DEVICE_RELATION_SOFT)
			continue;
		tmp = cd_profile_get_metadata_item (item->profile,
						    CD_PROFILE_METADATA_DATA_SOURCE);
		if (g_strcmp0 (tmp, CD_PROFILE_METADATA_DATA_SOURCE_EDID) == 0)
			continue;
		tmp = cd_profile_get_object_path (item->profile);
		profiles[idx++] = g_variant_new_object_path (tmp);
	}
	for (i = 0; i < priv->profiles->len; i++) {
		item = g_ptr_array_index (priv->profiles, i);
		if (item->relation != CD_DEVICE_RELATION_SOFT)
			continue;
		tmp = cd_profile_get_metadata_item (item->profile,
						    CD_PROFILE_METADATA_DATA_SOURCE);
		if (g_strcmp0 (tmp, CD_PROFILE_METADATA_DATA_SOURCE_EDID) != 0)
			continue;
		tmp = cd_profile_get_object_path (item->profile);
		profiles[idx++] = g_variant_new_object_path (tmp);
	}

	/* format the value */
	return g_variant_new_array (G_VARIANT_TYPE_OBJECT_PATH,
				    profiles,
				    priv->profiles->len);
}

gboolean
cd_device_remove_profile (CdDevice *device,
			  const gchar *profile_object_path,
			  GError **error)
{
	CdDevicePrivate *priv = GET_PRIVATE (device);
	CdDeviceProfileItem *item;
	gboolean ret = FALSE;
	guint i;

	/* check the profile exists on this device */
	for (i = 0; i < priv->profiles->len; i++) {
		item = g_ptr_array_index (priv->profiles, i);
		if (g_strcmp0 (profile_object_path,
			       cd_profile_get_object_path (item->profile)) == 0) {
			ret = TRUE;
			break;
		}
	}
	if (!ret) {
		g_set_error (error,
			     CD_DEVICE_ERROR,
			     CD_DEVICE_ERROR_PROFILE_DOES_NOT_EXIST,
			     "profile object path '%s' does not exist on '%s'",
			     profile_object_path,
			     priv->object_path);
		return FALSE;
	}

	/* remove from the arrays */
	ret = g_ptr_array_remove (priv->profiles, item);
	g_assert (ret);

	/* reset modification time */
	cd_device_reset_modified (device);

	/* emit */
	cd_device_dbus_emit_property_changed (device,
					      "Profiles",
					      cd_device_get_profiles_as_variant (device));

	/* emit global signal */
	cd_device_dbus_emit_device_changed (device);
	return TRUE;
}

static CdDeviceRelation
cd_device_find_profile_relation (CdDevice *device,
				 const gchar *profile_object_path)
{
	CdDevicePrivate *priv = GET_PRIVATE (device);
	CdDeviceProfileItem *item;
	guint i;

	/* search profiles */
	for (i = 0; i < priv->profiles->len; i++) {
		item = g_ptr_array_index (priv->profiles, i);
		if (g_strcmp0 (profile_object_path,
			       cd_profile_get_object_path (item->profile)) == 0) {
			return item->relation;
		}
	}
	return CD_DEVICE_RELATION_UNKNOWN;
}

static const gchar *
_cd_device_relation_to_string (CdDeviceRelation device_relation)
{
	if (device_relation == CD_DEVICE_RELATION_HARD)
		return "hard";
	if (device_relation == CD_DEVICE_RELATION_SOFT)
		return "soft";
	return "unknown";
}

static gint
cd_device_profile_item_sort_cb (gconstpointer a, gconstpointer b)
{
	gint64 tmp;
	CdDeviceProfileItem **item_a = (CdDeviceProfileItem **) a;
	CdDeviceProfileItem **item_b = (CdDeviceProfileItem **) b;

	tmp = (gint64) (*item_b)->timestamp - (gint64) (*item_a)->timestamp;
	if (tmp < 0)
		return -1;
	if (tmp > 0)
		return 1;
	return 0;
}

gboolean
cd_device_add_profile (CdDevice *device,
		       CdDeviceRelation relation,
		       const gchar *profile_object_path,
		       guint64 timestamp,
		       GError **error)
{
	CdDevicePrivate *priv = GET_PRIVATE (device);
	CdDeviceProfileItem *item;
	gboolean create_item = TRUE;
	guint i;
	g_autoptr(CdProfile) profile = NULL;

	/* is it available */
	profile = cd_profile_array_get_by_object_path (priv->profile_array,
						       profile_object_path);
	if (profile == NULL) {
		g_set_error (error,
			     CD_DEVICE_ERROR,
			     CD_DEVICE_ERROR_PROFILE_DOES_NOT_EXIST,
			     "profile object path '%s' does not exist",
			     profile_object_path);
		return FALSE;
	}

	/* check it does not already exist */
	for (i = 0; i < priv->profiles->len; i++) {
		item = g_ptr_array_index (priv->profiles, i);
		if (g_strcmp0 (cd_profile_get_object_path (profile),
			       cd_profile_get_object_path (item->profile)) == 0) {

			/* if we soft added this profile, and now the
			 * user hard adds it as well then we need to
			 * change the kind and not re-add it */
			if (relation == CD_DEVICE_RELATION_HARD &&
			    item->relation == CD_DEVICE_RELATION_SOFT) {
				g_debug ("CdDevice: converting %s hard->soft",
					 cd_profile_get_id (profile));
				item->relation = CD_DEVICE_RELATION_HARD;
				create_item = FALSE;
				break;
			}
			g_set_error (error,
				     CD_DEVICE_ERROR,
				     CD_DEVICE_ERROR_PROFILE_ALREADY_ADDED,
				     "profile object path '%s' has already been added",
				     profile_object_path);
			return FALSE;
		}
	}

	/* add to the array */
	if (create_item) {
		g_debug ("Adding %s [%s] to %s",
			 cd_profile_get_id (profile),
			 _cd_device_relation_to_string (relation),
			 priv->id);
		item = g_new0 (CdDeviceProfileItem, 1);
		item->profile = g_object_ref (profile);
		item->relation = relation;
		item->timestamp = timestamp;
		g_ptr_array_add (priv->profiles, item);
		g_ptr_array_sort (priv->profiles,
				  cd_device_profile_item_sort_cb);
	}

	/* reset modification time */
	cd_device_reset_modified (device);

	/* emit */
	cd_device_dbus_emit_property_changed (device,
					      "Profiles",
					      cd_device_get_profiles_as_variant (device));

	/* emit global signal */
	cd_device_dbus_emit_device_changed (device);
	return TRUE;
}

static void
cd_device_set_property_to_db (CdDevice *device,
			      const gchar *property,
			      const gchar *value)
{
	CdDevicePrivate *priv = GET_PRIVATE (device);
	gboolean ret;
	g_autoptr(GError) error = NULL;

	if (priv->object_scope != CD_OBJECT_SCOPE_DISK)
		return;

	ret = cd_device_db_set_property (priv->device_db,
					 priv->id,
					 property,
					 value,
					 &error);
	if (!ret) {
		g_warning ("CdDevice: failed to save property to database: %s",
			   error->message);
	}
}

static GVariant *
cd_device_get_metadata_as_variant (CdDevice *device)
{
	CdDevicePrivate *priv = GET_PRIVATE (device);
	GList *l;
	GVariantBuilder builder;
	g_autoptr(GList) list = NULL;

	/* do not try to build an empty array */
	if (g_hash_table_size (priv->metadata) == 0)
		return g_variant_new_array (G_VARIANT_TYPE ("{ss}"), NULL, 0);

	/* add all the keys in the dictionary to the variant builder */
	list = g_hash_table_get_keys (priv->metadata);
	g_variant_builder_init (&builder, G_VARIANT_TYPE_ARRAY);
	for (l = list; l != NULL; l = l->next) {
		g_variant_builder_add (&builder,
				       "{ss}",
				       l->data,
				       g_hash_table_lookup (priv->metadata,
							    l->data));
	}
	return g_variant_builder_end (&builder);
}

static void
cd_device_string_remove_suffix (gchar *vendor, const gchar *suffix)
{
	g_strchomp (vendor);
	if (g_str_has_suffix (vendor, suffix)) {
		gint len, suffix_len;
		len = strlen (vendor);
		suffix_len = strlen (suffix);
		vendor[len - suffix_len] = '\0';
	}
	g_strchomp (vendor);
}

static void
cd_device_set_vendor (CdDevice *device, const gchar *vendor)
{
	CdDevicePrivate *priv = GET_PRIVATE (device);
	g_free (priv->vendor);
	priv->vendor = cd_quirk_vendor_name (vendor);
}

static void
cd_device_set_model (CdDevice *device, const gchar *model)
{
	CdDevicePrivate *priv = GET_PRIVATE (device);
	GString *tmp;

	/* remove insanities */
	tmp = g_string_new (model);

	/* remove the kind suffix */
	if (priv->kind == CD_DEVICE_KIND_PRINTER)
		cd_device_string_remove_suffix (tmp->str, "Printer");
	if (priv->kind == CD_DEVICE_KIND_DISPLAY) {
		cd_device_string_remove_suffix (tmp->str, "Monitor");
		cd_device_string_remove_suffix (tmp->str, "Screen");
	}

	/* okay, we're done now */
	g_free (priv->model);
	priv->model = g_string_free (tmp, FALSE);
}

static GVariant *
cd_device_get_nullable_for_string (const gchar *value)
{
	if (value == NULL)
		return g_variant_new_string ("");
	return g_variant_new_string (value);
}

static void
cd_device_set_serial (CdDevice *device, const gchar *value)
{
	CdDevicePrivate *priv = GET_PRIVATE (device);
	gchar *tmp;

	/* CUPS likes to hand us a serial with a URI prepended */
	g_free (priv->serial);
	tmp = g_strstr_len (value, -1, "?serial=");
	if (tmp != NULL) {
		priv->serial = g_strdup (tmp + 8);
		return;
	}
	priv->serial = g_strdup (value);
}

gboolean
cd_device_set_property_internal (CdDevice *device,
				 const gchar *property,
				 const gchar *value,
				 gboolean save_in_db,
				 GError **error)
{
	gboolean is_metadata = FALSE;
	CdDevicePrivate *priv = GET_PRIVATE (device);

	/* sanity check the length of the key and value */
	if (strlen (property) > CD_DBUS_METADATA_KEY_LEN_MAX) {
		g_set_error_literal (error,
				     CD_CLIENT_ERROR,
				     CD_CLIENT_ERROR_INPUT_INVALID,
				     "metadata key length invalid");
		return FALSE;
	}
	if (value != NULL && strlen (value) > CD_DBUS_METADATA_VALUE_LEN_MAX) {
		g_set_error_literal (error,
				     CD_CLIENT_ERROR,
				     CD_CLIENT_ERROR_INPUT_INVALID,
				     "metadata value length invalid");
		return FALSE;
	}

	g_debug ("CdDevice: Attempting to set %s to %s on %s",
		 property, value, priv->id);
	if (g_strcmp0 (property, CD_DEVICE_PROPERTY_MODEL) == 0) {
		cd_device_set_model (device, value);
	} else if (g_strcmp0 (property, CD_DEVICE_PROPERTY_KIND) == 0) {
		priv->kind = cd_device_kind_from_string (value);
	} else if (g_strcmp0 (property, CD_DEVICE_PROPERTY_VENDOR) == 0) {
		cd_device_set_vendor (device, value);
	} else if (g_strcmp0 (property, CD_DEVICE_PROPERTY_SERIAL) == 0) {
		cd_device_set_serial (device, value);
	} else if (g_strcmp0 (property, CD_DEVICE_PROPERTY_COLORSPACE) == 0) {
		g_free (priv->colorspace);
		priv->colorspace = g_strdup (value);
	} else if (g_strcmp0 (property, CD_DEVICE_PROPERTY_FORMAT) == 0) {
		g_free (priv->format);
		priv->format = g_strdup (value);
	} else if (g_strcmp0 (property, CD_DEVICE_PROPERTY_MODE) == 0) {
		g_free (priv->mode);
		priv->mode = g_strdup (value);
	} else if (g_strcmp0 (property, CD_DEVICE_PROPERTY_SEAT) == 0) {
		g_free (priv->seat);
		priv->seat = g_strdup (value);
	} else if (g_strcmp0 (property, CD_DEVICE_PROPERTY_EMBEDDED) == 0) {
		priv->embedded = TRUE;
	} else {
		/* add to metadata */
		is_metadata = TRUE;
		g_hash_table_insert (priv->metadata,
				     g_strdup (property),
				     g_strdup (value));
		cd_device_dbus_emit_property_changed (device,
						      CD_DEVICE_PROPERTY_METADATA,
						      cd_device_get_metadata_as_variant (device));
	}

	/* set this externally so we can add disk devices at startup
	 * without re-adding */
	if (save_in_db) {
		cd_device_set_property_to_db (device,
					      property,
					      value);
	}

	/* if a known property, emit the correct property changed signal */
	if (!is_metadata) {
		cd_device_dbus_emit_property_changed (device,
						      property,
						      cd_device_get_nullable_for_string (value));
	}
	return TRUE;
}

const gchar *
cd_device_get_metadata (CdDevice *device, const gchar *key)
{
	CdDevicePrivate *priv = GET_PRIVATE (device);
	if (g_strcmp0 (key, CD_DEVICE_PROPERTY_MODEL) == 0)
		return priv->model;
	if (g_strcmp0 (key, CD_DEVICE_PROPERTY_VENDOR) == 0)
		return priv->vendor;
	if (g_strcmp0 (key, CD_DEVICE_PROPERTY_SERIAL) == 0)
		return priv->serial;
	return g_hash_table_lookup (priv->metadata, key);
}

gboolean
cd_device_make_default (CdDevice *device,
		        const gchar *profile_object_path,
		        GError **error)
{
	CdDevicePrivate *priv = GET_PRIVATE (device);
	CdDeviceProfileItem *item;
	CdProfile *profile;
	guint i;

	/* find profile */
	profile = cd_device_find_profile_by_object_path (priv->profiles,
							 profile_object_path);
	if (profile == NULL) {
		g_set_error (error,
			     CD_DEVICE_ERROR,
			     CD_DEVICE_ERROR_PROFILE_DOES_NOT_EXIST,
			     "profile object path '%s' does not exist for this device",
			     profile_object_path);
		return FALSE;
	}

	/* make the profile first in the array */
	for (i=1; i<priv->profiles->len; i++) {
		item = g_ptr_array_index (priv->profiles, i);
		if (item->profile == profile) {
			item->timestamp = g_get_real_time ();
			item->relation = CD_DEVICE_RELATION_HARD;
			g_ptr_array_sort (priv->profiles,
					  cd_device_profile_item_sort_cb);
			break;
		}
	}

	/* reset modification time */
	cd_device_reset_modified (device);

	/* emit */
	cd_device_dbus_emit_property_changed (device,
					      "Profiles",
					      cd_device_get_profiles_as_variant (device));

	/* emit global signal */
	cd_device_dbus_emit_device_changed (device);
	return TRUE;
}

static gboolean
cd_device_set_enabled (CdDevice *device,
		       gboolean enabled,
		       GError **error)
{
	CdDevicePrivate *priv = GET_PRIVATE (device);
	gboolean ret;
	g_autoptr(GError) error_local = NULL;

	/* device is already the correct state */
	if (priv->enabled == enabled)
		return TRUE;

	/* update database */
	ret = cd_device_db_set_property (priv->device_db,
					 priv->id,
					 "Enabled",
					 enabled ? "True" : "False",
					 &error_local);
	if (!ret) {
		g_set_error (error,
			     CD_DEVICE_ERROR,
			     CD_DEVICE_ERROR_INTERNAL,
			     "%s", error_local->message);
		return FALSE;
	}

	/* change property */
	priv->enabled = enabled;

	/* reset modification time */
	cd_device_reset_modified (device);

	/* emit */
	cd_device_dbus_emit_property_changed (device,
					      "Enabled",
					      g_variant_new_boolean (enabled));

	/* emit global signal */
	cd_device_dbus_emit_device_changed (device);
	return TRUE;
}

static void
cd_device_dbus_method_call (GDBusConnection *connection, const gchar *sender,
			    const gchar *object_path, const gchar *interface_name,
			    const gchar *method_name, GVariant *parameters,
			    GDBusMethodInvocation *invocation, gpointer user_data)
{
	CdDevice *device = CD_DEVICE (user_data);
	CdDevicePrivate *priv = GET_PRIVATE (device);
	CdDeviceRelation relation = CD_DEVICE_RELATION_UNKNOWN;
	CdProfile *profile = NULL;
	GVariant *tuple = NULL;
	GVariant *value = NULL;
	const gchar *id;
	const gchar *profile_object_path = NULL;
	const gchar *property_name = NULL;
	const gchar *property_value = NULL;
	gboolean ret;
	guint i = 0;
	g_autoptr(GError) error = NULL;

	/* return '' */
	if (g_strcmp0 (method_name, "AddProfile") == 0) {

		/* require auth */
		ret = cd_main_sender_authenticated (connection,
						    sender,
						    "org.freedesktop.color-manager.modify-device",
						    &error);
		if (!ret) {
			g_dbus_method_invocation_return_error (invocation,
							       CD_DEVICE_ERROR,
							       CD_DEVICE_ERROR_FAILED_TO_AUTHENTICATE,
							       "%s", error->message);
			return;
		}

		/* check the profile_object_path exists */
		g_variant_get (parameters, "(&s&o)",
			       &property_value,
			       &profile_object_path);
		g_debug ("CdDevice %s:AddProfile(%s)",
			 sender, profile_object_path);

		/* convert the device->profile relationship into an enum */
		if (g_strcmp0 (property_value, "soft") == 0)
			relation = CD_DEVICE_RELATION_SOFT;
		else if (g_strcmp0 (property_value, "hard") == 0)
			relation = CD_DEVICE_RELATION_HARD;

		/* nothing valid */
		if (relation == CD_DEVICE_RELATION_UNKNOWN) {
			g_dbus_method_invocation_return_error (invocation,
							       CD_DEVICE_ERROR,
							       CD_DEVICE_ERROR_INTERNAL,
							       "relation '%s' unknown, expected 'hard' or 'soft'",
							       property_value);
			return;
		}

		/* add it */
		ret = cd_device_add_profile (device,
					     relation,
					     profile_object_path,
					     g_get_real_time (),
					     &error);
		if (!ret) {
			g_dbus_method_invocation_return_gerror (invocation, error);
			return;
		}

		/* get profile id from object path */
		profile = cd_profile_array_get_by_object_path (priv->profile_array,
							       profile_object_path);
		id = cd_profile_get_id (profile);
		g_object_unref (profile);

		/* save this to the permanent database */
		if (relation == CD_DEVICE_RELATION_HARD) {
			ret = cd_mapping_db_add (priv->mapping_db,
						 priv->id,
						 id,
						 &error);
			if (!ret) {
				g_warning ("CdDevice: failed to save mapping to database: %s",
					   error->message);
			}
		}

		g_dbus_method_invocation_return_value (invocation, NULL);
		return;
	}

	if (g_strcmp0 (method_name, "RemoveProfile") == 0) {

		/* require auth */
		ret = cd_main_sender_authenticated (connection,
						    sender,
						    "org.freedesktop.color-manager.modify-device",
						    &error);
		if (!ret) {
			g_dbus_method_invocation_return_error (invocation,
							       CD_DEVICE_ERROR,
							       CD_DEVICE_ERROR_FAILED_TO_AUTHENTICATE,
							       "%s", error->message);
			return;
		}

		/* try to remove */
		g_variant_get (parameters, "(&o)",
			       &profile_object_path);
		g_debug ("CdDevice %s:RemoveProfile(%s)",
			 sender, profile_object_path);
		ret = cd_device_remove_profile (device,
						profile_object_path,
						&error);
		if (!ret) {
			g_dbus_method_invocation_return_gerror (invocation, error);
			return;
		}

		/* get profile id from object path */
		profile = cd_profile_array_get_by_object_path (priv->profile_array,
							       profile_object_path);
		id = cd_profile_get_id (profile);
		g_object_unref (profile);

		/* leave the entry in the database to it never gets
		 * soft added, even if there if metadata */
		ret = cd_mapping_db_clear_timestamp (priv->mapping_db,
						     priv->id,
						     id,
						     &error);
		if (!ret) {
			g_warning ("CdDevice: failed to save mapping to database: %s",
				   error->message);
		}

		g_dbus_method_invocation_return_value (invocation, NULL);
		return;
	}

	/* return 's' */
	if (g_strcmp0 (method_name, "GetProfileRelation") == 0) {

		/* find the profile relation */
		g_variant_get (parameters, "(o)", &property_value);
		g_debug ("CdDevice %s:GetProfileRelation(%s)",
			 sender, property_value);

		relation = cd_device_find_profile_relation (device,
							    property_value);
		if (relation == CD_DEVICE_RELATION_UNKNOWN) {
			g_dbus_method_invocation_return_error (invocation,
							       CD_DEVICE_ERROR,
							       CD_DEVICE_ERROR_PROFILE_DOES_NOT_EXIST,
							       "no profile '%s' found",
							       property_value);
			return;
		}

		tuple = g_variant_new ("(s)",
				       cd_device_relation_to_string (relation));
		g_dbus_method_invocation_return_value (invocation, tuple);
		return;
	}

	/* return 'o' */
	if (g_strcmp0 (method_name, "GetProfileForQualifiers") == 0) {
		gchar **regexes = NULL;
		g_autofree gchar *strv_debug = NULL;

		/* find the profile by the qualifier search string */
		g_variant_get (parameters, "(^a&s)", &regexes);

		/* show all the qualifiers */
		strv_debug = g_strjoinv (",", regexes);
		g_debug ("CdDevice %s:GetProfileForQualifiers(%s)",
			 sender, strv_debug);

		/* are we profiling? */
		ret = cd_inhibit_valid (priv->inhibit);
		if (!ret) {
			g_debug ("CdDevice: returning no results for profiling");
			g_dbus_method_invocation_return_error (invocation,
							       CD_DEVICE_ERROR,
							       CD_DEVICE_ERROR_PROFILING,
							       "profiling, so ignoring '%s'",
							       strv_debug);
			return;
		}

		/* search each regex against the profiles for this device */
		for (i = 0; profile == NULL && regexes[i] != NULL; i++) {
			if (i == 0)
				g_debug ("searching [hard]");
			profile = cd_device_find_by_qualifier (regexes[i],
							       priv->profiles,
							       CD_DEVICE_RELATION_HARD);
		}
		for (i = 0; profile == NULL && regexes[i] != NULL; i++) {
			if (i == 0)
				g_debug ("searching [soft]");
			profile = cd_device_find_by_qualifier (regexes[i],
							       priv->profiles,
							       CD_DEVICE_RELATION_SOFT);
		}
		if (profile == NULL) {
			g_dbus_method_invocation_return_error (invocation,
							       CD_DEVICE_ERROR,
							       CD_DEVICE_ERROR_NOTHING_MATCHED,
							       "nothing matched expression '%s'",
							       strv_debug);
			return;
		}

		value = g_variant_new_object_path (cd_profile_get_object_path (profile));
		tuple = g_variant_new_tuple (&value, 1);
		g_dbus_method_invocation_return_value (invocation, tuple);
		return;
	}

	/* return '' */
	if (g_strcmp0 (method_name, "MakeProfileDefault") == 0) {

		/* require auth */
		ret = cd_main_sender_authenticated (connection,
						    sender,
						    "org.freedesktop.color-manager.modify-device",
						    &error);
		if (!ret) {
			g_dbus_method_invocation_return_error (invocation,
							       CD_DEVICE_ERROR,
							       CD_DEVICE_ERROR_FAILED_TO_AUTHENTICATE,
							       "%s", error->message);
			return;
		}

		/* check the profile_object_path exists */
		g_variant_get (parameters, "(&o)",
			       &profile_object_path);
		g_debug ("CdDevice %s:MakeProfileDefault(%s)",
			 sender, profile_object_path);

		/* make profile default */
		ret = cd_device_make_default (device,
					      profile_object_path,
					      &error);
		if (!ret) {
			g_dbus_method_invocation_return_gerror (invocation, error);
			return;
		}

		/* reset modification time */
		cd_device_reset_modified (device);

		/* get profile id from object path */
		profile = cd_profile_array_get_by_object_path (priv->profile_array,
							       profile_object_path);
		id = cd_profile_get_id (profile);
		g_object_unref (profile);

		/* save new timestamp in database */
		ret = cd_mapping_db_add (priv->mapping_db,
					 priv->id,
					 id,
					 &error);
		if (!ret) {
			g_dbus_method_invocation_return_gerror (invocation, error);
			return;
		}

		g_dbus_method_invocation_return_value (invocation, NULL);
		return;
	}

	/* return '' */
	if (g_strcmp0 (method_name, "SetEnabled") == 0) {

		gboolean enabled;

		/* require auth */
		ret = cd_main_sender_authenticated (connection,
						    sender,
						    "org.freedesktop.color-manager.modify-device",
						    &error);
		if (!ret) {
			g_dbus_method_invocation_return_error (invocation,
							       CD_DEVICE_ERROR,
							       CD_DEVICE_ERROR_FAILED_TO_AUTHENTICATE,
							       "%s", error->message);
			return;
		}

		/* set, and parse */
		g_variant_get (parameters, "(b)",
			       &enabled);
		g_debug ("CdDevice %s:SetEnabled(%s)",
			 sender, enabled ? "True" : "False");
		ret = cd_device_set_enabled (device, enabled, &error);
		if (!ret) {
			g_dbus_method_invocation_return_gerror (invocation, error);
			return;
		}
		g_dbus_method_invocation_return_value (invocation, NULL);
		return;
	}

	/* return '' */
	if (g_strcmp0 (method_name, "SetProperty") == 0) {

		/* require auth */
		ret = cd_main_sender_authenticated (connection,
						    sender,
						    "org.freedesktop.color-manager.modify-device",
						    &error);
		if (!ret) {
			g_dbus_method_invocation_return_error (invocation,
							       CD_DEVICE_ERROR,
							       CD_DEVICE_ERROR_FAILED_TO_AUTHENTICATE,
							       "%s", error->message);
			return;
		}

		/* set, and parse */
		g_variant_get (parameters, "(&s&s)",
			       &property_name,
			       &property_value);
		g_debug ("CdDevice %s:SetProperty(%s,%s)",
			 sender, property_name, property_value);
		ret = cd_device_set_property_internal (device,
						       property_name,
						       property_value,
						       (priv->object_scope == CD_OBJECT_SCOPE_DISK),
						       &error);
		if (!ret) {
			g_dbus_method_invocation_return_gerror (invocation, error);
			return;
		}
		g_dbus_method_invocation_return_value (invocation, NULL);
		return;
	}

	/* return '' */
	if (g_strcmp0 (method_name, "ProfilingInhibit") == 0) {

		/* require auth */
		ret = cd_main_sender_authenticated (connection,
						    sender,
						    "org.freedesktop.color-manager.device-inhibit",
						    &error);
		if (!ret) {
			g_dbus_method_invocation_return_error (invocation,
							       CD_DEVICE_ERROR,
							       CD_DEVICE_ERROR_FAILED_TO_AUTHENTICATE,
							       "%s", error->message);
			return;
		}

		/* inhbit all profiles */
		g_debug ("CdDevice %s:ProfilingInhibit()",
			 sender);
		ret = cd_inhibit_add (priv->inhibit,
				      sender,
				      &error);
		if (!ret) {
			g_dbus_method_invocation_return_error (invocation,
							       CD_DEVICE_ERROR,
							       CD_DEVICE_ERROR_FAILED_TO_INHIBIT,
							       "%s", error->message);
			return;
		}
		g_dbus_method_invocation_return_value (invocation, NULL);
		return;
	}

	/* return '' */
	if (g_strcmp0 (method_name, "ProfilingUninhibit") == 0) {

		/* perhaps uninhibit all profiles */
		g_debug ("CdDevice %s:ProfilingUninhibit()",
			 sender);
		ret = cd_inhibit_remove (priv->inhibit,
					 sender,
					 &error);
		if (!ret) {
			g_dbus_method_invocation_return_error (invocation,
							       CD_DEVICE_ERROR,
							       CD_DEVICE_ERROR_FAILED_TO_UNINHIBIT,
							       "%s", error->message);
			return;
		}
		g_dbus_method_invocation_return_value (invocation, NULL);
		return;
	}

	/* we suck */
	g_critical ("failed to process device method %s", method_name);
}

static void
cd_device_inhibit_changed_cb (CdInhibit *inhibit,
			      gpointer user_data)
{
	CdDevice *device = CD_DEVICE (user_data);

	/* emit */
	g_debug ("Emitting Device.Profiles as inhibit changed");
	cd_device_dbus_emit_property_changed (device,
					      "Profiles",
					      cd_device_get_profiles_as_variant (device));

	/* emit global signal */
	cd_device_dbus_emit_device_changed (device);
}

static GVariant *
cd_device_dbus_get_property (GDBusConnection *connection_, const gchar *sender,
			     const gchar *object_path, const gchar *interface_name,
			     const gchar *property_name, GError **error,
			     gpointer user_data)
{
	CdDevice *device = CD_DEVICE (user_data);
	CdDevicePrivate *priv = GET_PRIVATE (device);
	g_auto(GStrv) bus_names = NULL;

	if (g_strcmp0 (property_name, CD_DEVICE_PROPERTY_CREATED) == 0)
		return g_variant_new_uint64 (priv->created);
	if (g_strcmp0 (property_name, CD_DEVICE_PROPERTY_MODIFIED) == 0)
		return g_variant_new_uint64 (priv->modified);
	if (g_strcmp0 (property_name, CD_DEVICE_PROPERTY_MODEL) == 0)
		return cd_device_get_nullable_for_string (priv->model);
	if (g_strcmp0 (property_name, CD_DEVICE_PROPERTY_VENDOR) == 0)
		return cd_device_get_nullable_for_string (priv->vendor);
	if (g_strcmp0 (property_name, CD_DEVICE_PROPERTY_SERIAL) == 0)
		return cd_device_get_nullable_for_string (priv->serial);
	if (g_strcmp0 (property_name, CD_DEVICE_PROPERTY_ENABLED) == 0)
		return g_variant_new_boolean (priv->enabled);
	if (g_strcmp0 (property_name, CD_DEVICE_PROPERTY_COLORSPACE) == 0)
		return cd_device_get_nullable_for_string (priv->colorspace);
	if (g_strcmp0 (property_name, CD_DEVICE_PROPERTY_FORMAT) == 0)
		return cd_device_get_nullable_for_string (priv->format);
	if (g_strcmp0 (property_name, CD_DEVICE_PROPERTY_MODE) == 0)
		return cd_device_get_nullable_for_string (priv->mode);
	if (g_strcmp0 (property_name, CD_DEVICE_PROPERTY_KIND) == 0)
		return cd_device_get_nullable_for_string (cd_device_kind_to_string (priv->kind));
	if (g_strcmp0 (property_name, CD_DEVICE_PROPERTY_ID) == 0)
		return g_variant_new_string (priv->id);
	if (g_strcmp0 (property_name, CD_DEVICE_PROPERTY_PROFILES) == 0)
		return cd_device_get_profiles_as_variant (device);
	if (g_strcmp0 (property_name, CD_DEVICE_PROPERTY_METADATA) == 0)
		return cd_device_get_metadata_as_variant (device);
	if (g_strcmp0 (property_name, CD_DEVICE_PROPERTY_SCOPE) == 0)
		return g_variant_new_string (cd_object_scope_to_string (priv->object_scope));
	if (g_strcmp0 (property_name, CD_DEVICE_PROPERTY_OWNER) == 0)
		return g_variant_new_uint32 (priv->owner);
	if (g_strcmp0 (property_name, CD_DEVICE_PROPERTY_SEAT) == 0)
		return cd_device_get_nullable_for_string (priv->seat);
	if (g_strcmp0 (property_name, CD_DEVICE_PROPERTY_EMBEDDED) == 0)
		return g_variant_new_boolean (priv->embedded);
	if (g_strcmp0 (property_name, CD_DEVICE_PROPERTY_PROFILING_INHIBITORS) == 0) {
		bus_names = cd_inhibit_get_bus_names (priv->inhibit);
		return g_variant_new_strv ((const gchar * const *) bus_names, -1);
	}

	/* return an error */
	g_set_error (error,
		     CD_DEVICE_ERROR,
		     CD_DEVICE_ERROR_INTERNAL,
		     "failed to get device property %s",
		     property_name);
	return NULL;
}

gboolean
cd_device_register_object (CdDevice *device,
			   GDBusConnection *connection,
			   GDBusInterfaceInfo *info,
			   GError **error)
{
	CdDevicePrivate *priv = GET_PRIVATE (device);
	g_autoptr(GError) error_local = NULL;

	static const GDBusInterfaceVTable interface_vtable = {
		cd_device_dbus_method_call,
		cd_device_dbus_get_property,
		NULL
	};

	priv->connection = connection;
	priv->registration_id = g_dbus_connection_register_object (
		connection,
		priv->object_path,
		info,
		&interface_vtable,
		device,  /* user_data */
		NULL,  /* user_data_free_func */
		&error_local); /* GError** */
	if (priv->registration_id == 0) {
		g_set_error (error,
			     CD_DEVICE_ERROR,
			     CD_DEVICE_ERROR_INTERNAL,
			     "failed to register object: %s",
			     error_local->message);
		return FALSE;
	}
	g_debug ("CdDevice: Register interface %u on %s",
		 priv->registration_id,
		 priv->object_path);
	return TRUE;
}

static void
cd_device_name_vanished_cb (GDBusConnection *connection,
			    const gchar *name,
			    gpointer user_data)
{
	CdDevice *device = CD_DEVICE (user_data);
	g_debug ("CdDevice: emit 'invalidate' as %s vanished", name);
	g_signal_emit (device, signals[SIGNAL_INVALIDATE], 0);
}

void
cd_device_watch_sender (CdDevice *device, const gchar *sender)
{
	CdDevicePrivate *priv = GET_PRIVATE (device);
	g_return_if_fail (CD_IS_DEVICE (device));
	g_return_if_fail (sender != NULL);
	priv->watcher_id = g_bus_watch_name (G_BUS_TYPE_SYSTEM,
						     sender,
						     G_BUS_NAME_WATCHER_FLAGS_NONE,
						     NULL,
						     cd_device_name_vanished_cb,
						     device,
						     NULL);
}

static void
cd_device_get_property (GObject *object, guint prop_id, GValue *value, GParamSpec *pspec)
{
	CdDevice *device = CD_DEVICE (object);
	CdDevicePrivate *priv = GET_PRIVATE (device);

	switch (prop_id) {
	case PROP_OBJECT_PATH:
		g_value_set_string (value, priv->object_path);
		break;
	case PROP_ID:
		g_value_set_string (value, priv->id);
		break;
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
		break;
	}
}

static void
cd_device_set_property (GObject *object, guint prop_id, const GValue *value, GParamSpec *pspec)
{
	CdDevice *device = CD_DEVICE (object);
	CdDevicePrivate *priv = GET_PRIVATE (device);

	switch (prop_id) {
	case PROP_OBJECT_PATH:
		g_free (priv->object_path);
		priv->object_path = g_strdup (g_value_get_string (value));
		break;
	case PROP_ID:
		g_free (priv->id);
		priv->id = g_strdup (g_value_get_string (value));
		break;
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
		break;
	}
}

static void
cd_device_class_init (CdDeviceClass *klass)
{
	GParamSpec *pspec;
	GObjectClass *object_class = G_OBJECT_CLASS (klass);
	object_class->finalize = cd_device_finalize;
	object_class->get_property = cd_device_get_property;
	object_class->set_property = cd_device_set_property;

	/**
	 * CdDevice:object-path:
	 */
	pspec = g_param_spec_string ("object-path", NULL, NULL,
				     NULL,
				     G_PARAM_READWRITE);
	g_object_class_install_property (object_class, PROP_OBJECT_PATH, pspec);

	/**
	 * CdDevice:id:
	 */
	pspec = g_param_spec_string ("id", NULL, NULL,
				     NULL,
				     G_PARAM_READWRITE);
	g_object_class_install_property (object_class, PROP_ID, pspec);

	/**
	 * CdDevice::invalidate:
	 **/
	signals[SIGNAL_INVALIDATE] =
		g_signal_new ("invalidate",
			      G_TYPE_FROM_CLASS (object_class), G_SIGNAL_RUN_LAST,
			      G_STRUCT_OFFSET (CdDeviceClass, invalidate),
			      NULL, NULL, g_cclosure_marshal_VOID__VOID,
			      G_TYPE_NONE, 0);
}

static void
cd_device_profiles_item_free (CdDeviceProfileItem *item)
{
	g_object_unref (item->profile);
	g_free (item);
}

static void
cd_device_init (CdDevice *device)
{
	CdDevicePrivate *priv = GET_PRIVATE (device);
	priv->profiles = g_ptr_array_new_with_free_func ((GDestroyNotify) cd_device_profiles_item_free);
	priv->profile_array = cd_profile_array_new ();
	priv->created = g_get_real_time ();
	priv->modified = g_get_real_time ();
	priv->mapping_db = cd_mapping_db_new ();
	priv->device_db = cd_device_db_new ();
	priv->inhibit = cd_inhibit_new ();
	g_signal_connect (priv->inhibit,
			  "changed",
			  G_CALLBACK (cd_device_inhibit_changed_cb),
			  device);
	priv->metadata = g_hash_table_new_full (g_str_hash,
							 g_str_equal,
							 g_free,
							 g_free);
}

static void
cd_device_finalize (GObject *object)
{
	CdDevice *device = CD_DEVICE (object);
	CdDevicePrivate *priv = GET_PRIVATE (device);

	if (priv->watcher_id > 0)
		g_bus_unwatch_name (priv->watcher_id);
	if (priv->registration_id > 0) {
		g_debug ("CdDevice: Unregister interface %u on %s",
			  priv->registration_id,
			  priv->object_path);
		g_dbus_connection_unregister_object (priv->connection,
						     priv->registration_id);
	}
	g_free (priv->id);
	g_free (priv->model);
	g_free (priv->vendor);
	g_free (priv->colorspace);
	g_free (priv->format);
	g_free (priv->mode);
	g_free (priv->serial);
	g_free (priv->seat);
	g_free (priv->object_path);
	g_ptr_array_unref (priv->profiles);
	g_object_unref (priv->profile_array);
	g_object_unref (priv->mapping_db);
	g_object_unref (priv->device_db);
	g_object_unref (priv->inhibit);
	g_hash_table_unref (priv->metadata);

	G_OBJECT_CLASS (cd_device_parent_class)->finalize (object);
}

CdDevice *
cd_device_new (void)
{
	CdDevice *device;
	device = g_object_new (CD_TYPE_DEVICE, NULL);
	return CD_DEVICE (device);
}
