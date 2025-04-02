#include "ait.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <gtk/gtk.h>
#include <glade/glade.h>
#include <glib.h>
#include <glib/gprintf.h>
#include <sys/wait.h>

/*
static int entry_has_text(const char * entry_name)
{
    GtkEntry * entry;
    entry = GTK_ENTRY(glade_xml_get_widget(glade_xml, entry_name));
    return strlen(gtk_entry_get_text(entry)) > 0;
}
*/

static const gchar * double_to_string(double value)
{
    static gchar buf[32];

    /* in this context, NAN means "not specified", so leave blank */
    if (ISNAN(value))
        return "";

    snprintf(buf, sizeof(buf), "%lf", value);
    return buf;
}

const char * datum_string(int datum)
{
    switch (datum)
    {
        default:
            assert(FALSE);
            return "";

        case WGS84_DATUM:
            return "WGS84";

        case NAD27_DATUM:
            return "NAD27";

        case NAD83_DATUM:
            return "NAD83";
    }
}

const char * resample_method_string(resample_method_t resample_method)
{
    switch (resample_method)
    {
        case RESAMPLE_NEAREST_NEIGHBOR:
            return "Nearest Neighbor";
            break;

        default:
            assert(FALSE);
            return "";

        case RESAMPLE_BILINEAR:
            return "Bilinear";
            break;
    
        case RESAMPLE_BICUBIC:
            return "Bicubic";
            break;
    }
}

void geocode_options_changed()
{
    int average_height_is_checked;
    int pixel_size_is_checked;

    int enable_projection_option_menu = FALSE;
    int enable_predefined_projection_option_menu = FALSE;

    int enable_utm_zone = FALSE;
    int enable_central_meridian = FALSE;
    int enable_latitude_of_origin = FALSE;
    int enable_first_standard_parallel = FALSE;
    int enable_second_standard_parallel = FALSE;

    int enable_height_checkbutton = FALSE;
    int enable_height_entry = FALSE;
    int enable_pixel_size_checkbutton = FALSE;
    int enable_pixel_size_entry = FALSE;
    int enable_force_checkbutton = FALSE;
    int enable_resample_optionmenu = FALSE;
    int enable_datum_optionmenu = FALSE;

    GtkWidget *geocode_checkbutton =
        glade_xml_get_widget(glade_xml, "geocode_checkbutton");

    int geocode_projection_is_checked =
        gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(geocode_checkbutton));

    GtkWidget *height_checkbutton =
        glade_xml_get_widget(glade_xml, "height_checkbutton");

    GtkWidget *pixel_size_checkbutton =
        glade_xml_get_widget(glade_xml, "pixel_size_checkbutton");

    GtkWidget *projection_option_menu =
        glade_xml_get_widget(glade_xml, "projection_optionmenu");

    GtkWidget *predefined_projection_option_menu =
        glade_xml_get_widget(glade_xml, "predefined_projection_optionmenu");

    GtkWidget *zone_entry =
        glade_xml_get_widget(glade_xml, "zone_entry");

    GtkWidget *central_meridian_entry =
        glade_xml_get_widget(glade_xml, "central_meridian_entry");

    GtkWidget *latitude_of_origin_entry =
        glade_xml_get_widget(glade_xml, "latitude_of_origin_entry");

    GtkWidget *first_standard_parallel_entry =
        glade_xml_get_widget(glade_xml, "first_standard_parallel_entry");

    GtkWidget *second_standard_parallel_entry =
        glade_xml_get_widget(glade_xml, "second_standard_parallel_entry");

    int projection =
        gtk_option_menu_get_history(GTK_OPTION_MENU(projection_option_menu));

    int enable_utm_projection_options = projection == PROJ_UTM;

    if (geocode_projection_is_checked)
    {	
        int predefined_projection_is_selected =
            0 < gtk_option_menu_get_history(
               GTK_OPTION_MENU(predefined_projection_option_menu));

        enable_projection_option_menu = TRUE;
        enable_predefined_projection_option_menu = TRUE;

        if (predefined_projection_is_selected)
        {
            /* all widgets remain disabled -- load settings from file */
            project_parameters_t * pps =
                load_selected_predefined_projection_parameters(projection);

            if (!pps)
            {
                predefined_projection_is_selected = FALSE;
            }
            else
            {
                gtk_entry_set_text(
                    GTK_ENTRY(zone_entry), "");
                gtk_entry_set_text(
                    GTK_ENTRY(central_meridian_entry), "");
                gtk_entry_set_text(
                    GTK_ENTRY(latitude_of_origin_entry), "");
                gtk_entry_set_text(
                    GTK_ENTRY(first_standard_parallel_entry), "");
                gtk_entry_set_text(
                    GTK_ENTRY(second_standard_parallel_entry), "");

                switch (projection)
                {
                case PROJ_UTM:
                    // no UTM predefined projections
                    // we shouldn't be here
                    assert(FALSE);
                    break;

                case PROJ_PS:
                    gtk_entry_set_text(
                        GTK_ENTRY(central_meridian_entry),
                        double_to_string(pps->ps.slon));
                    gtk_entry_set_text(
                        GTK_ENTRY(first_standard_parallel_entry),
                        double_to_string(pps->ps.slat));
                    break;

                case PROJ_LAMCC:
                    gtk_entry_set_text(
                        GTK_ENTRY(first_standard_parallel_entry),
                        double_to_string(pps->lamcc.plat1));
                    gtk_entry_set_text(
                        GTK_ENTRY(second_standard_parallel_entry),
                        double_to_string(pps->lamcc.plat2));
                    gtk_entry_set_text(
                        GTK_ENTRY(central_meridian_entry),
                        double_to_string(pps->lamcc.lon0));
                    gtk_entry_set_text(
                        GTK_ENTRY(latitude_of_origin_entry),
                        double_to_string(pps->lamcc.lat0));
                    break;

                case PROJ_LAMAZ:
                    gtk_entry_set_text(
                        GTK_ENTRY(central_meridian_entry),
                        double_to_string(pps->lamaz.center_lon));
                    gtk_entry_set_text(
                        GTK_ENTRY(latitude_of_origin_entry),
                        double_to_string(pps->lamaz.center_lat));
                    break;

                case PROJ_ALBERS:
                    gtk_entry_set_text(
                        GTK_ENTRY(first_standard_parallel_entry),
                        double_to_string(pps->albers.std_parallel1));
                    gtk_entry_set_text(
                        GTK_ENTRY(second_standard_parallel_entry),
                        double_to_string(pps->albers.std_parallel2));
                    gtk_entry_set_text(
                        GTK_ENTRY(central_meridian_entry),
                        double_to_string(pps->albers.center_meridian));
                    gtk_entry_set_text(
                        GTK_ENTRY(latitude_of_origin_entry),
                        double_to_string(pps->albers.orig_latitude));
                    break;
                }

                free(pps);
            }
        }

        if (!predefined_projection_is_selected)
        {	    
            switch (projection)
            {
            case PROJ_UTM:
                enable_utm_zone = TRUE;
                break;

            case PROJ_PS:
                enable_central_meridian = TRUE;
                enable_first_standard_parallel = TRUE;

                gtk_entry_set_text(
                    GTK_ENTRY(latitude_of_origin_entry), "");
                gtk_entry_set_text(
                    GTK_ENTRY(second_standard_parallel_entry), "");

                break;

            case PROJ_LAMCC:
                enable_first_standard_parallel = TRUE;
                enable_second_standard_parallel = TRUE;
                enable_central_meridian = TRUE;
                enable_latitude_of_origin = TRUE;
                break;

            case PROJ_LAMAZ:
                enable_central_meridian = TRUE;
                enable_latitude_of_origin = TRUE;

                gtk_entry_set_text(
                    GTK_ENTRY(first_standard_parallel_entry), "");
                gtk_entry_set_text(
                    GTK_ENTRY(second_standard_parallel_entry), "");

                break;

            case PROJ_ALBERS:
                enable_first_standard_parallel = TRUE;
                enable_second_standard_parallel = TRUE;
                enable_central_meridian = TRUE;
                enable_latitude_of_origin = TRUE;
                break;
            }
        }

        enable_pixel_size_checkbutton = TRUE;
        enable_height_checkbutton = TRUE;
        enable_datum_optionmenu = TRUE;
        enable_resample_optionmenu = TRUE;
        enable_force_checkbutton = TRUE;

        average_height_is_checked = 
            gtk_toggle_button_get_active(
                GTK_TOGGLE_BUTTON(height_checkbutton));

        pixel_size_is_checked = 
            gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(
            pixel_size_checkbutton));

        if (average_height_is_checked)
            enable_height_entry = TRUE;

        if (pixel_size_is_checked)
            enable_pixel_size_entry = TRUE;

        set_predefined_projections(projection);
    }

    gtk_widget_set_sensitive(projection_option_menu,
        enable_projection_option_menu);

    gtk_widget_set_sensitive(predefined_projection_option_menu,
        enable_predefined_projection_option_menu &&
        !enable_utm_projection_options);

    GtkWidget *zone_label =
        glade_xml_get_widget(glade_xml, "zone_label");

    GtkWidget *central_meridian_label =
        glade_xml_get_widget(glade_xml, "central_meridian_label");

    GtkWidget *latitude_of_origin_label =
        glade_xml_get_widget(glade_xml, "latitude_of_origin_label");

    GtkWidget *first_standard_parallel_label =
        glade_xml_get_widget(glade_xml, "first_standard_parallel_label");

    GtkWidget *second_standard_parallel_label =
        glade_xml_get_widget(glade_xml, "second_standard_parallel_label");

    GtkWidget *central_meridian_hbox =
        glade_xml_get_widget(glade_xml, "central_meridian_hbox");

    GtkWidget *latitude_of_origin_hbox =
        glade_xml_get_widget(glade_xml, "latitude_of_origin_hbox");

    GtkWidget *first_standard_parallel_hbox =
        glade_xml_get_widget(glade_xml, "first_standard_parallel_hbox");

    GtkWidget *second_standard_parallel_hbox =
        glade_xml_get_widget(glade_xml, "second_standard_parallel_hbox");

    GtkWidget *height_hbox =
        glade_xml_get_widget(glade_xml, "height_hbox");

    GtkWidget *pixel_size_hbox =
        glade_xml_get_widget(glade_xml, "pixel_size_hbox");

    GtkWidget *datum_optionmenu =
        glade_xml_get_widget(glade_xml, "datum_optionmenu");

    GtkWidget *datum_label =
        glade_xml_get_widget(glade_xml, "datum_label");

    GtkWidget *resample_optionmenu =
        glade_xml_get_widget(glade_xml, "resample_optionmenu");

    GtkWidget *resample_label =
        glade_xml_get_widget(glade_xml, "resample_label");

    GtkWidget *force_checkbutton =
        glade_xml_get_widget(glade_xml, "force_checkbutton");

    gtk_widget_set_sensitive(zone_entry,
        enable_utm_zone);

    gtk_widget_set_sensitive(zone_label,
        enable_utm_zone);

    gtk_widget_set_sensitive(central_meridian_hbox, 
        enable_central_meridian);

    gtk_widget_set_sensitive(central_meridian_label,
        enable_central_meridian);

    gtk_widget_set_sensitive(latitude_of_origin_hbox,
        enable_latitude_of_origin);

    gtk_widget_set_sensitive(latitude_of_origin_label,
        enable_latitude_of_origin);

    gtk_widget_set_sensitive(first_standard_parallel_hbox,
        enable_first_standard_parallel);

    gtk_widget_set_sensitive(first_standard_parallel_label,
        enable_first_standard_parallel);

    gtk_widget_set_sensitive(second_standard_parallel_hbox,
        enable_second_standard_parallel);

    gtk_widget_set_sensitive(second_standard_parallel_label,
        enable_second_standard_parallel);

    gtk_widget_set_sensitive(height_checkbutton,
        enable_height_checkbutton);

    gtk_widget_set_sensitive(height_hbox,
        enable_height_entry);

    gtk_widget_set_sensitive(pixel_size_checkbutton,
        enable_pixel_size_checkbutton);

    gtk_widget_set_sensitive(pixel_size_hbox,
        enable_pixel_size_entry);

    gtk_widget_set_sensitive(datum_optionmenu,
        enable_datum_optionmenu);

    gtk_widget_set_sensitive(datum_label,
        enable_datum_optionmenu);

    gtk_widget_set_sensitive(resample_optionmenu,
        enable_resample_optionmenu);

    gtk_widget_set_sensitive(resample_label,
        enable_resample_optionmenu);

    gtk_widget_set_sensitive(force_checkbutton,
        enable_force_checkbutton);

    if (enable_utm_projection_options)
    {
        gtk_widget_show(zone_label);
        gtk_widget_show(zone_entry);

        gtk_widget_hide(central_meridian_label);
        gtk_widget_hide(central_meridian_hbox);

        gtk_widget_hide(latitude_of_origin_label);
        gtk_widget_hide(latitude_of_origin_hbox);

        gtk_widget_hide(first_standard_parallel_label);
        gtk_widget_hide(first_standard_parallel_hbox);

        gtk_widget_hide(second_standard_parallel_label);
        gtk_widget_hide(second_standard_parallel_hbox);
    }
    else
    {
        gtk_widget_hide(zone_label);
        gtk_widget_hide(zone_entry);

        gtk_widget_show(central_meridian_label);
        gtk_widget_show(central_meridian_hbox);

        gtk_widget_show(latitude_of_origin_label);
        gtk_widget_show(latitude_of_origin_hbox);

        gtk_widget_show(first_standard_parallel_label);
        gtk_widget_show(first_standard_parallel_hbox);

        gtk_widget_show(second_standard_parallel_label);
        gtk_widget_show(second_standard_parallel_hbox);
    }

    update_summary();
}

SIGNAL_CALLBACK void
on_albers_conical_equal_area_activate(GtkWidget * widget)
{
    geocode_options_changed();
}

SIGNAL_CALLBACK void
on_lambert_conformal_conic_activate(GtkWidget * widget)
{
    geocode_options_changed();
}

SIGNAL_CALLBACK void
on_lambert_azimuthal_equal_area_activate(GtkWidget * widget)
{
    geocode_options_changed();
}

SIGNAL_CALLBACK void
on_polar_stereographic_activate(GtkWidget * widget)
{
    geocode_options_changed();
}

SIGNAL_CALLBACK void
on_utm_activate(GtkWidget * widget)
{
    geocode_options_changed();
}

SIGNAL_CALLBACK void
on_geocode_checkbutton_toggled(GtkWidget * widget)
{
    geocode_options_changed();
}

SIGNAL_CALLBACK void
on_height_checkbutton_toggled(GtkWidget * widget)
{
    geocode_options_changed();
}

SIGNAL_CALLBACK void
on_pixel_size_checkbutton_toggled(GtkWidget * widget)
{
    geocode_options_changed();
}

SIGNAL_CALLBACK void
on_wgs84_activate(GtkWidget * widget)
{
    geocode_options_changed();
}

SIGNAL_CALLBACK void
on_nad27_activate(GtkWidget * widget)
{
    geocode_options_changed();
}

SIGNAL_CALLBACK void
on_nad83_activate(GtkWidget * widget)
{
    geocode_options_changed();
}

SIGNAL_CALLBACK void
on_nearest_neighbor_activate(GtkWidget *widget)
{
    geocode_options_changed();
}

SIGNAL_CALLBACK void
on_bilinear_activate(GtkWidget *widget)
{
    geocode_options_changed();
}

SIGNAL_CALLBACK void
on_bicubic_activate(GtkWidget *widget)
{
    geocode_options_changed();
}


SIGNAL_CALLBACK void
on_predefined_projection_optionmenu_changed(GtkWidget * widget)
{
    geocode_options_changed();
}

SIGNAL_CALLBACK void
on_user_defined_activate(GtkWidget * widget)
{
}
