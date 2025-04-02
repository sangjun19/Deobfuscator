// Repository: opendlang/opend
// File: libraries/upstream/cairoD/src/cairo/c/cairo.d

/**
 * Cairo C API. Contains stuff from cairo.h and cairo-version.h
 *
 * This module only contains basic documentation. For more information
 * see $(LINK http://cairographics.org/manual/)
 *
 * License:
 * $(TABLE
 *   $(TR $(TD cairoD wrapper/bindings)
 *     $(TD $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)))
 *   $(TR $(TD $(LINK2 http://cgit.freedesktop.org/cairo/tree/COPYING, _cairo))
 *     $(TD $(LINK2 http://cgit.freedesktop.org/cairo/tree/COPYING-LGPL-2.1, LGPL 2.1) /
 *     $(LINK2 http://cgit.freedesktop.org/cairo/plain/COPYING-MPL-1.1, MPL 1.1)))
 * )
 * Authors:
 * $(TABLE
 *   $(TR $(TD Johannes Pfau) $(TD cairoD))
 *   $(TR $(TD $(LINK2 http://cairographics.org, _cairo team)) $(TD _cairo))
 * )
 */
/*
 * Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 */
module cairo.c.cairo;

import std.conv;
public import cairo.c.config;

/**
 * Cairo binding version. Use the cairo_version() function to get
 * version information about the cairo library.
 */
enum CAIRO_VERSION_MAJOR = 1;
///ditto
enum CAIRO_VERSION_MINOR = 10;
///ditto
enum CAIRO_VERSION_MICRO = 2;

///
ulong CAIRO_VERSION_ENCODE(uint major, uint minor, uint micro)
{
    return (major * 10000) + (minor * 100) + (micro);
}

///Encoded cairo binding version
enum ulong CAIRO_VERSION = CAIRO_VERSION_ENCODE(CAIRO_VERSION_MAJOR,
    CAIRO_VERSION_MINOR, CAIRO_VERSION_MICRO);

///
string CAIRO_VERSION_STRINGIZE(uint major, uint minor, uint micro)
{
    return to!string(major) ~ "." ~ to!string(minor) ~ "." ~ to!string(micro);
}

///Cairo binding version string
string CAIRO_VERSION_STRING = CAIRO_VERSION_STRINGIZE(CAIRO_VERSION_MAJOR,
    CAIRO_VERSION_MINOR, CAIRO_VERSION_MICRO);


extern(C)
{
    ///Encoded library version
    int cairo_version();

    ///Library version string
    immutable(char)* cairo_version_string();

    /**
     * cairo_bool_t is used for boolean values. Returns of type
     * cairo_bool_t will always be either 0 or 1, but testing against
     * these values explicitly is not encouraged; just use the
     * value as a boolean condition.
     *
     * Examples:
     * ----------------------------------------
     *  if (cairo_in_stroke (cr, x, y)) {
     *      //do something
     *  }
     * ----------------------------------------
     **/
    alias int cairo_bool_t;

    /**
     * A cairo_t contains the current state of the rendering device,
     * including coordinates of yet to be drawn shapes.
     *
     * Cairo contexts, as cairo_t objects are named, are central to
     * cairo and all drawing with cairo is always done to a cairo_t
     * object.
     *
     * Memory management of cairo_t is done with
     * cairo_reference() and cairo_destroy().
     **/
    struct cairo_t {};

    /**
     * A cairo_surface_t represents an image, either as the destination
     * of a drawing operation or as source when drawing onto another
     * surface.  To draw to a cairo_surface_t, create a cairo context
     * with the surface as the target, using cairo_create().
     *
     * There are different subtypes of cairo_surface_t for
     * different drawing backends; for example, cairo_image_surface_create()
     * creates a bitmap image in memory.
     * The type of a surface can be queried with cairo_surface_get_type().
     *
     * The initial contents of a surface after creation depend upon the manner
     * of its creation. If cairo creates the surface and backing storage for
     * the user, it will be initially cleared; for example,
     * cairo_image_surface_create() and cairo_surface_create_similar().
     * Alternatively, if the user passes in a reference to some backing storage
     * and asks cairo to wrap that in a cairo_surface_t, then the contents are
     * not modified; for example, cairo_image_surface_create_for_data() and
     * cairo_xlib_surface_create().
     *
     * Memory management of cairo_surface_t is done with
     * cairo_surface_reference() and cairo_surface_destroy().
     **/
    struct cairo_surface_t {};

    /**
     * A cairo_device_t represents the driver interface for drawing
     * operations to a cairo_surface_t.  There are different subtypes of
     * cairo_device_t for different drawing backends; for example,
     * cairo_xcb_device_create() creates a device that wraps the connection
     * to an X Windows System using the XCB library.
     *
     * The type of a device can be queried with cairo_device_get_type().
     *
     * Memory management of cairo_device_t is done with
     * cairo_device_reference() and cairo_device_destroy().
     *
     * Since: 1.10
     **/
    struct cairo_device_t {};

    /**
     * A $(D cairo_matrix_t) holds an affine transformation, such as a scale,
     * rotation, shear, or a combination of those. The transformation of
     * a point (x, y) is given by:
     * --------------------------------------
     *     x_new = xx * x + xy * y + x0;
     *     y_new = yx * x + yy * y + y0;
     * --------------------------------------
     **/
    struct cairo_matrix_t
    {
        double xx; ///xx component of the affine transformation
        double yx; ///yx component of the affine transformation
        double xy; ///xy component of the affine transformation
        double yy; ///yy component of the affine transformation
        double x0; ///X translation component of the affine transformation
        double y0; ///Y translation component of the affine transformation
    }

    /**
     * A $(D cairo_pattern_t) represents a source when drawing onto a
     * surface. There are different subtypes of $(D cairo_pattern_t),
     * for different types of sources; for example,
     * $(D cairo_pattern_create_rgb()) creates a pattern for a solid
     * opaque color.
     *
     * Other than various cairo_pattern_create_$(B type)()
     * functions, some of the pattern types can be implicitly created
     * using various cairo_set_source_$(B type)() functions;
     * for example cairo_set_source_rgb().
     *
     * The type of a pattern can be queried with cairo_pattern_get_type().
     *
     * Memory management of $(D cairo_pattern_t) is done with
     * cairo_pattern_reference() and cairo_pattern_destroy().
     **/
    struct cairo_pattern_t {};

    /**
     * $(D cairo_destroy_func_t) the type of function which is called when a
     * data element is destroyed. It is passed the pointer to the data
     * element and should free any memory and resources allocated for it.
     *
     * Params:
     * data = The data element being destroyed.
     **/
    alias extern(C) void function(void* data) cairo_destroy_func_t;

    /**
     * $(D cairo_user_data_key_t) is used for attaching user data to cairo
     * data structures.  The actual contents of the struct is never used,
     * and there is no need to initialize the object; only the unique
     * address of a $(D cairo_data_key_t) object is used.  Typically, you
     * would just use the address of a static $(D cairo_data_key_t) object.
     **/
    struct cairo_user_data_key_t
    {
        ///not used; ignore.
        int unused;
    }

    /**

     * $(D cairo_status_t) is used to indicate errors that can occur when
     * using Cairo. In some cases it is returned directly by functions.
     * but when using $(D cairo_t), the last error, if any, is stored in
     * the context and can be retrieved with cairo_status().
     *
     * New entries may be added in future versions.  Use cairo_status_to_string()
     * to get a human-readable representation of an error message.
     **/
    enum cairo_status_t
    {
        CAIRO_STATUS_SUCCESS = 0, ///no error has occurred

        CAIRO_STATUS_NO_MEMORY, ///out of memory
        CAIRO_STATUS_INVALID_RESTORE, ///cairo_restore() called without matching cairo_save()
        CAIRO_STATUS_INVALID_POP_GROUP, ///no saved group to pop, i.e. cairo_pop_group() without matching cairo_push_group()
        CAIRO_STATUS_NO_CURRENT_POINT, ///no current point defined
        CAIRO_STATUS_INVALID_MATRIX, ///invalid matrix (not invertible)
        CAIRO_STATUS_INVALID_STATUS, ///invalid value for an input $(D cairo_status_t)
        CAIRO_STATUS_NULL_POINTER, ///$(D null) pointer
        CAIRO_STATUS_INVALID_STRING, ///input string not valid UTF-8
        CAIRO_STATUS_INVALID_PATH_DATA, ///input path data not valid
        CAIRO_STATUS_READ_ERROR, ///error while reading from input stream
        CAIRO_STATUS_WRITE_ERROR, ///error while writing to output stream
        CAIRO_STATUS_SURFACE_FINISHED, ///target surface has been finished
        CAIRO_STATUS_SURFACE_TYPE_MISMATCH, ///the surface type is not appropriate for the operation
        CAIRO_STATUS_PATTERN_TYPE_MISMATCH, ///the pattern type is not appropriate for the operation
        CAIRO_STATUS_INVALID_CONTENT, ///invalid value for an input $(D cairo_content_t)
        CAIRO_STATUS_INVALID_FORMAT, ///invalid value for an input $(D cairo_format_t)
        CAIRO_STATUS_INVALID_VISUAL, ///invalid value for an input Visual*
        CAIRO_STATUS_FILE_NOT_FOUND, ///file not found
        CAIRO_STATUS_INVALID_DASH, ///invalid value for a dash setting
        CAIRO_STATUS_INVALID_DSC_COMMENT, ///invalid value for a DSC comment (Since 1.2)
        CAIRO_STATUS_INVALID_INDEX, ///invalid index passed to getter (Since 1.4)
        CAIRO_STATUS_CLIP_NOT_REPRESENTABLE, ///clip region not representable in desired format (Since 1.4)
        CAIRO_STATUS_TEMP_FILE_ERROR, ///error creating or writing to a temporary file (Since 1.6)
        CAIRO_STATUS_INVALID_STRIDE, ///invalid value for stride (Since 1.6)
        CAIRO_STATUS_FONT_TYPE_MISMATCH, ///the font type is not appropriate for the operation (Since 1.8)
        CAIRO_STATUS_USER_FONT_IMMUTABLE, ///the user-font is immutable (Since 1.8)
        CAIRO_STATUS_USER_FONT_ERROR, ///error occurred in a user-font callback function (Since 1.8)
        CAIRO_STATUS_NEGATIVE_COUNT, ///negative number used where it is not allowed (Since 1.8)
        CAIRO_STATUS_INVALID_CLUSTERS, ///input clusters do not represent the accompanying text and glyph array (Since 1.8)
        CAIRO_STATUS_INVALID_SLANT, ///invalid value for an input $(D cairo_font_slant_t) (Since 1.8)
        CAIRO_STATUS_INVALID_WEIGHT, ///invalid value for an input $(D cairo_font_weight_t) (Since 1.8)
        CAIRO_STATUS_INVALID_SIZE, ///invalid value (typically too big) for the size of the input (surface, pattern, etc.) (Since 1.10)
        CAIRO_STATUS_USER_FONT_NOT_IMPLEMENTED, ///user-font method not implemented (Since 1.10)
        CAIRO_STATUS_DEVICE_TYPE_MISMATCH, ///the device type is not appropriate for the operation (Since 1.10)
        CAIRO_STATUS_DEVICE_ERROR, ///an operation to the device caused an unspecified error (Since 1.10)
        /**
         * this is a special value indicating the number of
         * status values defined in this enumeration.  When using this value, note
         * that the version of cairo at run-time may have additional status values
         * defined than the value of this symbol at compile-time. (Since 1.10)
         */
        CAIRO_STATUS_LAST_STATUS
    }

    /**
     * $(D cairo_content_t) is used to describe the content that a surface will
     * contain, whether color information, alpha information (translucence
     * vs. opacity), or both.
     *
     * Note: The large values here are designed to keep $(D cairo_content_t)
     * values distinct from $(D cairo_format_t) values so that the
     * implementation can detect the error if users confuse the two types.
     **/
    enum cairo_content_t
    {
        CAIRO_CONTENT_COLOR = 0x1000, ///The surface will hold color content only.
        CAIRO_CONTENT_ALPHA = 0x2000, ///The surface will hold alpha content only.
        CAIRO_CONTENT_COLOR_ALPHA = 0x3000 ///The surface will hold color and alpha content.
    }

    /**
     * $(D cairo_write_func_t) is the type of function which is called when a
     * backend needs to write data to an output stream.  It is passed the
     * closure which was specified by the user at the time the write
     * function was registered, the data to write and the length of the
     * data in bytes.  The write function should return
     * $(D CAIRO_STATUS_SUCCESS) if all the data was successfully written,
     * $(D CAIRO_STATUS_WRITE_ERROR) otherwise.
     *
     * Params:
     * closure = the output closure
     * data = the buffer containing the data to write
     * length = the amount of data to write
     *
     * Returns: the status code of the write operation
     **/
    alias extern(C) cairo_status_t function(void* closure, const ubyte* data, uint length) cairo_write_func_t;

    /**
     * $(D cairo_read_func_t) is the type of function which is called when a
     * backend needs to read data from an input stream.  It is passed the
     * closure which was specified by the user at the time the read
     * function was registered, the buffer to read the data into and the
     * length of the data in bytes.  The read function should return
     * $(D CAIRO_STATUS_SUCCESS) if all the data was successfully read,
     * $(D CAIRO_STATUS_READ_ERROR) otherwise.
     *
     * Params:
     * closure = the input closure
     * data = the buffer into which to read the data
     * length = the amount of data to read
     *
     * Returns: the status code of the read operation
     **/
    alias extern(C) cairo_status_t function(void* closure, ubyte* data, uint length) cairo_read_func_t;

     /** Functions for manipulating state objects */
     cairo_t* cairo_create(cairo_surface_t* target);
     ///ditto
     cairo_t* cairo_reference(cairo_t* cr);
     ///ditto
     void cairo_destroy(cairo_t* cr);
     ///ditto
     uint cairo_get_reference_count(cairo_t* cr);
     ///ditto
     void* cairo_get_user_data(cairo_t* cr, const cairo_user_data_key_t* key);
     ///ditto
     cairo_status_t cairo_set_user_data (cairo_t* cr, const cairo_user_data_key_t* key,
        void* user_data,
        cairo_destroy_func_t destroy);
     ///ditto
     void cairo_save(cairo_t* cr);
     ///ditto
     void cairo_restore(cairo_t* cr);
     ///ditto
     void cairo_push_group(cairo_t* cr);
     ///ditto
     void cairo_push_group_with_content(cairo_t* cr, cairo_content_t content);
     ///ditto
     cairo_pattern_t* cairo_pop_group(cairo_t* cr);
     ///ditto
     void cairo_pop_group_to_source(cairo_t* cr);

    /* Modify state */

    /**
     * $(D cairo_operator_t) is used to set the compositing operator for all cairo
     * drawing operations.
     *
     * The default operator is $(D CAIRO_OPERATOR_OVER).
     *
     * The operators marked as $(I unbounded) modify their
     * destination even outside of the mask layer (that is, their effect is not
     * bound by the mask layer).  However, their effect can still be limited by
     * way of clipping.
     *
     * To keep things simple, the operator descriptions here
     * document the behavior for when both source and destination are either fully
     * transparent or fully opaque.  The actual implementation works for
     * translucent layers too.
     * For a more detailed explanation of the effects of each operator, including
     * the mathematical definitions, see
     * $(LINK http://cairographics.org/operators/).
     **/
    enum cairo_operator_t
    {
        CAIRO_OPERATOR_CLEAR,///clear destination layer (bounded)

        CAIRO_OPERATOR_SOURCE,///replace destination layer (bounded)
        CAIRO_OPERATOR_OVER,///draw source layer on top of destination layer (bounded)
        CAIRO_OPERATOR_IN,///draw source where there was destination content (unbounded)
        CAIRO_OPERATOR_OUT,///draw source where there was no destination content (unbounded)
        CAIRO_OPERATOR_ATOP,///draw source on top of destination content and only there

        CAIRO_OPERATOR_DEST,///ignore the source
        CAIRO_OPERATOR_DEST_OVER,///draw destination on top of source
        CAIRO_OPERATOR_DEST_IN,///leave destination only where there was source content (unbounded)
        CAIRO_OPERATOR_DEST_OUT,///leave destination only where there was no source content
        CAIRO_OPERATOR_DEST_ATOP,///leave destination on top of source content and only there (unbounded)

        CAIRO_OPERATOR_XOR,///source and destination are shown where there is only one of them
        CAIRO_OPERATOR_ADD,///source and destination layers are accumulated
        CAIRO_OPERATOR_SATURATE,///like over, but assuming source and dest are disjoint geometries

        CAIRO_OPERATOR_MULTIPLY,///source and destination layers are multiplied. This causes the result to be at least as dark as the darker inputs.
        CAIRO_OPERATOR_SCREEN,///source and destination are complemented and multiplied. This causes the result to be at least as light as the lighter inputs.
        CAIRO_OPERATOR_OVERLAY,///multiplies or screens, depending on the lightness of the destination color.
        CAIRO_OPERATOR_DARKEN,///replaces the destination with the source if it is darker, otherwise keeps the source.
        CAIRO_OPERATOR_LIGHTEN,///replaces the destination with the source if it is lighter, otherwise keeps the source.
        CAIRO_OPERATOR_COLOR_DODGE,///brightens the destination color to reflect the source color.
        CAIRO_OPERATOR_COLOR_BURN,///darkens the destination color to reflect the source color.
        CAIRO_OPERATOR_HARD_LIGHT,///Multiplies or screens, dependant on source color.
        CAIRO_OPERATOR_SOFT_LIGHT,///Darkens or lightens, dependant on source color.
        CAIRO_OPERATOR_DIFFERENCE,///Takes the difference of the source and destination color.
        CAIRO_OPERATOR_EXCLUSION,///Produces an effect similar to difference, but with lower contrast.
        CAIRO_OPERATOR_HSL_HUE,///Creates a color with the hue of the source and the saturation and luminosity of the target.
        CAIRO_OPERATOR_HSL_SATURATION,///Creates a color with the saturation of the source and the hue and luminosity of the target. Painting with this mode onto a gray area prduces no change.
        /**
         * Creates a color with the hue and saturation
         * of the source and the luminosity of the target. This preserves the gray
         * levels of the target and is useful for coloring monochrome images or
         * tinting color images.
         */
        CAIRO_OPERATOR_HSL_COLOR,
        /**
         * Creates a color with the luminosity of
         * the source and the hue and saturation of the target. This produces an
         * inverse effect to $(D CAIRO_OPERATOR_HSL_COLOR).
         */
        CAIRO_OPERATOR_HSL_LUMINOSITY
    }
     /** Modify state */
     void cairo_set_operator(cairo_t* cr, cairo_operator_t op);
     ///ditto
     void cairo_set_source (cairo_t* cr, cairo_pattern_t* source);
     ///ditto
     void cairo_set_source_rgb(cairo_t* cr, double red, double green, double blue);
     ///ditto
     void
    cairo_set_source_rgba (cairo_t* cr,
                   double red, double green, double blue,
                   double alpha);
     ///ditto
     void
    cairo_set_source_surface (cairo_t* cr,
                  cairo_surface_t* surface,
                  double	   x,
                  double	   y);
     ///ditto
     void
    cairo_set_tolerance (cairo_t* cr, double tolerance);

    /**
     * Specifies the type of antialiasing to do when rendering text or shapes.
     **/
    enum cairo_antialias_t
    {
        CAIRO_ANTIALIAS_DEFAULT, ///Use the default antialiasing for the subsystem and target device
        CAIRO_ANTIALIAS_NONE, ///Use a bilevel alpha mask
        /**
         * Perform single-color antialiasing (using
         * shades of gray for black text on a white background, for example).
         */
        CAIRO_ANTIALIAS_GRAY,
        /**
         * Perform antialiasing by taking
         * advantage of the order of subpixel elements on devices
         * such as LCD panels
         */
        CAIRO_ANTIALIAS_SUBPIXEL
    }
    ///
     void
    cairo_set_antialias (cairo_t* cr, cairo_antialias_t antialias);

    /**
     * $(D cairo_fill_rule_t) is used to select how paths are filled. For both
     * fill rules, whether or not a point is included in the fill is
     * determined by taking a ray from that point to infinity and looking
     * at intersections with the path. The ray can be in any direction,
     * as long as it doesn't pass through the end point of a segment
     * or have a tricky intersection such as intersecting tangent to the path.
     * (Note that filling is not actually implemented in this way. This
     * is just a description of the rule that is applied.)
     *
     * The default fill rule is $(D CAIRO_FILL_RULE_WINDING).
     *
     * New entries may be added in future versions.
     **/
    enum cairo_fill_rule_t
    {
        /**
         * If the path crosses the ray from
         * left-to-right, counts +1. If the path crosses the ray
         * from right to left, counts -1. (Left and right are determined
         * from the perspective of looking along the ray from the starting
         * point.) If the total count is non-zero, the point will be filled.
         */
        CAIRO_FILL_RULE_WINDING,
        /**
         * Counts the total number of
         * intersections, without regard to the orientation of the contour. If
         * the total number of intersections is odd, the point will be
         * filled.
         */
        CAIRO_FILL_RULE_EVEN_ODD
    }
    ///
     void
    cairo_set_fill_rule (cairo_t* cr, cairo_fill_rule_t fill_rule);
    ///
     void
    cairo_set_line_width (cairo_t* cr, double width);

    /**
     * Specifies how to render the endpoints of the path when stroking.
     *
     * The default line cap style is $(D CAIRO_LINE_CAP_BUTT).
     **/
    enum cairo_line_cap_t
    {
        CAIRO_LINE_CAP_BUTT, ///start(stop) the line exactly at the start(end) point
        CAIRO_LINE_CAP_ROUND, ///use a round ending, the center of the circle is the end point
        CAIRO_LINE_CAP_SQUARE ///use squared ending, the center of the square is the end point
    }
    ///
     void
    cairo_set_line_cap (cairo_t* cr, cairo_line_cap_t line_cap);

    /**
     * Specifies how to render the junction of two lines when stroking.
     *
     * The default line join style is $(D CAIRO_LINE_JOIN_MITER).
     **/
    enum cairo_line_join_t
    {
        CAIRO_LINE_JOIN_MITER, ///use a sharp (angled) corner, see cairo_set_miter_limit()
        /**
         * use a rounded join, the center of the circle is the
         * joint point
         */
        CAIRO_LINE_JOIN_ROUND,
        /**
         * use a cut-off join, the join is cut off at half
         * the line width from the joint point
         */
        CAIRO_LINE_JOIN_BEVEL
    }
    ///
     void
    cairo_set_line_join (cairo_t* cr, cairo_line_join_t line_join);
    ///
     void
    cairo_set_dash (cairo_t      *cr,
            const double *dashes,
            int	      num_dashes,
            double	      offset);
    ///
     void
    cairo_set_miter_limit (cairo_t* cr, double limit);
    ///
     void
    cairo_translate (cairo_t* cr, double tx, double ty);
    ///
     void
    cairo_scale (cairo_t* cr, double sx, double sy);
    ///
     void
    cairo_rotate (cairo_t* cr, double angle);
    ///
     void
    cairo_transform (cairo_t	      *cr,
             const cairo_matrix_t *matrix);
    ///
     void
    cairo_set_matrix (cairo_t	       *cr,
              const cairo_matrix_t *matrix);
    ///
     void
    cairo_identity_matrix (cairo_t* cr);
    ///
     void
    cairo_user_to_device (cairo_t* cr, double *x, double *y);
    ///
     void
    cairo_user_to_device_distance (cairo_t* cr, double *dx, double *dy);
    ///
     void
    cairo_device_to_user (cairo_t* cr, double *x, double *y);
    ///
     void
    cairo_device_to_user_distance (cairo_t* cr, double *dx, double *dy);

    /** Path creation functions */
     void
    cairo_new_path (cairo_t* cr);
    ///ditto
     void
    cairo_move_to (cairo_t* cr, double x, double y);
    ///ditto
     void
    cairo_new_sub_path (cairo_t* cr);
    ///ditto
     void
    cairo_line_to (cairo_t* cr, double x, double y);
    ///ditto
     void
    cairo_curve_to (cairo_t* cr,
            double x1, double y1,
            double x2, double y2,
            double x3, double y3);
    ///ditto
     void
    cairo_arc (cairo_t* cr,
           double xc, double yc,
           double radius,
           double angle1, double angle2);
    ///ditto
     void
    cairo_arc_negative (cairo_t* cr,
                double xc, double yc,
                double radius,
                double angle1, double angle2);

    /* XXX: NYI
     void
    cairo_arc_to (cairo_t* cr,
              double x1, double y1,
              double x2, double y2,
              double radius);
    */
    ///ditto
     void
    cairo_rel_move_to (cairo_t* cr, double dx, double dy);
    ///ditto
     void
    cairo_rel_line_to (cairo_t* cr, double dx, double dy);
    ///ditto
     void
    cairo_rel_curve_to (cairo_t* cr,
                double dx1, double dy1,
                double dx2, double dy2,
                double dx3, double dy3);
    ///ditto
     void
    cairo_rectangle (cairo_t* cr,
             double x, double y,
             double width, double height);

    /* XXX: NYI
     void
    cairo_stroke_to_path (cairo_t* cr);
    */
    ///ditto
     void
    cairo_close_path (cairo_t* cr);
    ///ditto
     void
    cairo_path_extents (cairo_t* cr,
                double *x1, double *y1,
                double *x2, double *y2);

    /** Painting functions */
     void
    cairo_paint (cairo_t* cr);
    ///ditto
     void
    cairo_paint_with_alpha (cairo_t* cr,
                double   alpha);
    ///ditto
     void
    cairo_mask (cairo_t         *cr,
            cairo_pattern_t *pattern);
    ///ditto
     void
    cairo_mask_surface (cairo_t         *cr,
                cairo_surface_t *surface,
                double           surface_x,
                double           surface_y);
    ///ditto
     void
    cairo_stroke (cairo_t* cr);
    ///ditto
     void
    cairo_stroke_preserve (cairo_t* cr);
    ///ditto
     void
    cairo_fill (cairo_t* cr);
    ///ditto
     void
    cairo_fill_preserve (cairo_t* cr);
    ///ditto
     void
    cairo_copy_page (cairo_t* cr);
    ///ditto
     void
    cairo_show_page (cairo_t* cr);

    /** Insideness testing */
     cairo_bool_t
    cairo_in_stroke (cairo_t* cr, double x, double y);
    ///ditto
     cairo_bool_t
    cairo_in_fill (cairo_t* cr, double x, double y);
    ///ditto
     cairo_bool_t
    cairo_in_clip (cairo_t* cr, double x, double y);

    /** Rectangular extents */
     void
    cairo_stroke_extents (cairo_t* cr,
                  double *x1, double *y1,
                  double *x2, double *y2);
    ///ditto
     void
    cairo_fill_extents (cairo_t* cr,
                double *x1, double *y1,
                double *x2, double *y2);

    /** Clipping */
     void
    cairo_reset_clip (cairo_t* cr);
    ///ditto
     void
    cairo_clip (cairo_t* cr);
    ///ditto
     void
    cairo_clip_preserve (cairo_t* cr);
    ///ditto
     void
    cairo_clip_extents (cairo_t* cr,
                double *x1, double *y1,
                double *x2, double *y2);

    /**
     * A data structure for holding a rectangle.
     *
     * Since: 1.4
     **/
    struct cairo_rectangle_t
    {
        double x; ///X coordinate of the left side of the rectangle
        double y; ///Y coordinate of the the top side of the rectangle
        double width; ///width of the rectangle
        double height; ///height of the rectangle
    }

    /**
     * A data structure for holding a dynamically allocated
     * array of rectangles.
     *
     * Since: 1.4
     **/
    struct cairo_rectangle_list_t
    {
        cairo_status_t     status; ///Error status of the rectangle list
        cairo_rectangle_t *rectangles; ///Array containing the rectangles
        int                num_rectangles; ///Number of rectangles in this list
    }
    ///
     cairo_rectangle_list_t *
    cairo_copy_clip_rectangle_list (cairo_t* cr);
    ///
     void
    cairo_rectangle_list_destroy (cairo_rectangle_list_t *rectangle_list);

    /* Font/Text functions */

    /**
     * A $(D cairo_scaled_font_t) is a font scaled to a particular size and device
     * resolution. A $(D cairo_scaled_font_t) is most useful for low-level font
     * usage where a library or application wants to cache a reference
     * to a scaled font to speed up the computation of metrics.
     *
     * There are various types of scaled fonts, depending on the
     * $(I font backend) they use. The type of a
     * scaled font can be queried using cairo_scaled_font_get_type().
     *
     * Memory management of $(D cairo_scaled_font_t) is done with
     * cairo_scaled_font_reference() and cairo_scaled_font_destroy().
     **/
    struct cairo_scaled_font_t {};

    /**
     * A $(D cairo_font_face_t) specifies all aspects of a font other
     * than the size or font matrix (a font matrix is used to distort
     * a font by sheering it or scaling it unequally in the two
     * directions) . A font face can be set on a $(D cairo_t) by using
     * cairo_set_font_face(); the size and font matrix are set with
     * cairo_set_font_size() and cairo_set_font_matrix().
     *
     * There are various types of font faces, depending on the
     * $(I font backend) they use. The type of a
     * font face can be queried using cairo_font_face_get_type().
     *
     * Memory management of $(D cairo_font_face_t) is done with
     * cairo_font_face_reference() and cairo_font_face_destroy().
     **/
    struct cairo_font_face_t {};

    /**
     * The $(D cairo_glyph_t) structure holds information about a single glyph
     * when drawing or measuring text. A font is (in simple terms) a
     * collection of shapes used to draw text. A glyph is one of these
     * shapes. There can be multiple glyphs for a single character
     * (alternates to be used in different contexts, for example), or a
     * glyph can be a $(I ligature) of multiple
     * characters. Cairo doesn't expose any way of converting input text
     * into glyphs, so in order to use the Cairo interfaces that take
     * arrays of glyphs, you must directly access the appropriate
     * underlying font system.
     *
     * Note that the offsets given by $(D x) and $(D y) are not cumulative. When
     * drawing or measuring text, each glyph is individually positioned
     * with respect to the overall origin
     **/
    struct cairo_glyph_t
    {
        /**
         * glyph index in the font. The exact interpretation of the
         * glyph index depends on the font technology being used.
         */
        ulong        index;
        /**
         * the offset in the X direction between the origin used for
         * drawing or measuring the string and the origin of this glyph.
         */
        double               x;
        /**
         * the offset in the Y direction between the origin used for
         * drawing or measuring the string and the origin of this glyph.
         */
        double               y;
    }
    ///
     cairo_glyph_t *
    cairo_glyph_allocate (int num_glyphs);
    ///
     void
    cairo_glyph_free (cairo_glyph_t *glyphs);

    /**
     * The $(D cairo_text_cluster_t) structure holds information about a single
     * $(I text cluster).  A text cluster is a minimal
     * mapping of some glyphs corresponding to some UTF-8 text.
     *
     * For a cluster to be valid, both $(D num_bytes) and $(D num_glyphs) should
     * be non-negative, and at least one should be non-zero.
     * Note that clusters with zero glyphs are not as well supported as
     * normal clusters.  For example, PDF rendering applications typically
     * ignore those clusters when PDF text is being selected.
     *
     * See cairo_show_text_glyphs() for how clusters are used in advanced
     * text operations.
     *
     * Since: 1.8
     **/
    struct cairo_text_cluster_t
    {
        int        num_bytes; ///the number of bytes of UTF-8 text covered by cluster
        int        num_glyphs; ///the number of glyphs covered by cluster
    }
    ///
     cairo_text_cluster_t *
    cairo_text_cluster_allocate (int num_clusters);
    ///
     void
    cairo_text_cluster_free (cairo_text_cluster_t *clusters);

    /**
     * Specifies properties of a text cluster mapping.
     *
     * Since: 1.8
     **/
    enum cairo_text_cluster_flags_t
    {
        /**
         * The clusters in the cluster array
         * map to glyphs in the glyph array from end to start.
         */
        CAIRO_TEXT_CLUSTER_FLAG_BACKWARD = 0x00000001
    }

    /**
     * The $(D cairo_text_extents_t) structure stores the extents of a single
     * glyph or a string of glyphs in user-space coordinates. Because text
     * extents are in user-space coordinates, they are mostly, but not
     * entirely, independent of the current transformation matrix. If you call
     * $(D cairo_scale(cr, 2.0, 2.0)), text will
     * be drawn twice as big, but the reported text extents will not be
     * doubled. They will change slightly due to hinting (so you can't
     * assume that metrics are independent of the transformation matrix),
     * but otherwise will remain unchanged.
     **/
    struct cairo_text_extents_t
    {
        /**
         * the horizontal distance from the origin to the
         * leftmost part of the glyphs as drawn. Positive if the
         * glyphs lie entirely to the right of the origin.
         */
        double x_bearing;
        /**
         * the vertical distance from the origin to the
         * topmost part of the glyphs as drawn. Positive only if the
         * glyphs lie completely below the origin; will usually be
         * negative.
         */
        double y_bearing;
        /**
         * width of the glyphs as drawn
         */
        double width;
        /**
         * height of the glyphs as drawn
         */
        double height;
        /**
         * distance to advance in the X direction
         * after drawing these glyphs
         */
        double x_advance;
        /**
         * distance to advance in the Y direction
         * after drawing these glyphs. Will typically be zero except
         * for vertical text layout as found in East-Asian languages.
         */
        double y_advance;
    }

    /**
     * The $(D cairo_font_extents_t) structure stores metric information for
     * a font. Values are given in the current user-space coordinate
     * system.
     *
     * Because font metrics are in user-space coordinates, they are
     * mostly, but not entirely, independent of the current transformation
     * matrix. If you call $(D cairo_scale(cr, 2.0, 2.0)),
     * text will be drawn twice as big, but the reported text extents will
     * not be doubled. They will change slightly due to hinting (so you
     * can't assume that metrics are independent of the transformation
     * matrix), but otherwise will remain unchanged.
     **/
    struct cairo_font_extents_t
    {
        /**
         * the distance that the font extends above the baseline.
         * Note that this is not always exactly equal to the maximum
         * of the extents of all the glyphs in the font, but rather
         * is picked to express the font designer's intent as to
         * how the font should align with elements above it.
         */
        double ascent;
        /**
         * the distance that the font extends below the baseline.
         *  This value is positive for typical fonts that include
         *  portions below the baseline. Note that this is not always
         *  exactly equal to the maximum of the extents of all the
         *  glyphs in the font, but rather is picked to express the
         *  font designer's intent as to how the the font should
         *  align with elements below it.
         */
        double descent;
        /**
         * the recommended vertical distance between baselines when
         * setting consecutive lines of text with the font. This
         * is greater than @ascent+@descent by a
         * quantity known as the $(I line spacing)
         * or $(I external leading). When space
         * is at a premium, most fonts can be set with only
         * a distance of @ascent+@descent between lines.
         */
        double height;
        /**
         * the maximum distance in the X direction that
         *         the the origin is advanced for any glyph in the font.
         */
        double max_x_advance;
        /**
         * the maximum distance in the Y direction that
         *         the the origin is advanced for any glyph in the font.
         *         this will be zero for normal fonts used for horizontal
         *         writing. (The scripts of East Asia are sometimes written
         *         vertically.)
         */
        double max_y_advance;
    }

    /**
     * Specifies variants of a font face based on their slant.
     **/
    enum cairo_font_slant_t
    {
        CAIRO_FONT_SLANT_NORMAL, ///Upright font style
        CAIRO_FONT_SLANT_ITALIC, ///Italic font style
        CAIRO_FONT_SLANT_OBLIQUE ///Oblique font style
    }

    /**
     * Specifies variants of a font face based on their weight.
     **/
    enum cairo_font_weight_t
    {
        CAIRO_FONT_WEIGHT_NORMAL, ///Normal font weight
        CAIRO_FONT_WEIGHT_BOLD ///Bold font weight
    }

    /**
     * The subpixel order specifies the order of color elements within
     * each pixel on the display device when rendering with an
     * antialiasing mode of $(D CAIRO_ANTIALIAS_SUBPIXEL).
     **/
    enum cairo_subpixel_order_t
    {
        /**
         * Use the default subpixel order for
         * for the target device
         */
        CAIRO_SUBPIXEL_ORDER_DEFAULT,
        /**
         * Subpixel elements are arranged horizontally
         * with red at the left
         */
        CAIRO_SUBPIXEL_ORDER_RGB,
        /**
         * Subpixel elements are arranged horizontally
         * with blue at the left
         */
        CAIRO_SUBPIXEL_ORDER_BGR,
        /**
         * Subpixel elements are arranged vertically
         * with red at the top
         */
        CAIRO_SUBPIXEL_ORDER_VRGB,
        /**
         * Subpixel elements are arranged vertically
         * with blue at the top
         */
        CAIRO_SUBPIXEL_ORDER_VBGR
    }

    /**
     * Specifies the type of hinting to do on font outlines. Hinting
     * is the process of fitting outlines to the pixel grid in order
     * to improve the appearance of the result. Since hinting outlines
     * involves distorting them, it also reduces the faithfulness
     * to the original outline shapes. Not all of the outline hinting
     * styles are supported by all font backends.
     *
     * New entries may be added in future versions.
     **/
    enum cairo_hint_style_t
    {
        CAIRO_HINT_STYLE_DEFAULT, ///Use the default hint style for font backend and target device
        CAIRO_HINT_STYLE_NONE, ///Do not hint outlines
        /**
         * Hint outlines slightly to improve
         * contrast while retaining good fidelity to the original
         * shapes.
         */
        CAIRO_HINT_STYLE_SLIGHT,
        /**
         * Hint outlines with medium strength
         * giving a compromise between fidelity to the original shapes
         * and contrast
         */
        CAIRO_HINT_STYLE_MEDIUM,
        CAIRO_HINT_STYLE_FULL ///Hint outlines to maximize contrast
    }

    /**
     * Specifies whether to hint font metrics; hinting font metrics
     * means quantizing them so that they are integer values in
     * device space. Doing this improves the consistency of
     * letter and line spacing, however it also means that text
     * will be laid out differently at different zoom factors.
     **/
    enum cairo_hint_metrics_t
    {
        /**
         * Hint metrics in the default
         * manner for the font backend and target device
         */
        CAIRO_HINT_METRICS_DEFAULT,
        CAIRO_HINT_METRICS_OFF, ///Do not hint font metrics
        CAIRO_HINT_METRICS_ON ///Hint font metrics
    }

    /**
     * An opaque structure holding all options that are used when
     * rendering fonts.
     *
     * Individual features of a $(D cairo_font_options_t) can be set or
     * accessed using functions named
     * cairo_font_options_set_$(B feature_name) and
     * cairo_font_options_get_$(B feature_name), like
     * cairo_font_options_set_antialias() and
     * cairo_font_options_get_antialias().
     *
     * New features may be added to a $(D cairo_font_options_t) in the
     * future.  For this reason, cairo_font_options_copy(),
     * cairo_font_options_equal(), cairo_font_options_merge(), and
     * cairo_font_options_hash() should be used to copy, check
     * for equality, merge, or compute a hash value of
     * $(D cairo_font_options_t) objects.
     **/
    struct cairo_font_options_t {};
    ///
     cairo_font_options_t *
    cairo_font_options_create ();
    ///
     cairo_font_options_t *
    cairo_font_options_copy (const cairo_font_options_t *original);
    ///
     void
    cairo_font_options_destroy (cairo_font_options_t *options);
    ///
     cairo_status_t
    cairo_font_options_status (cairo_font_options_t *options);
    ///
     void
    cairo_font_options_merge (cairo_font_options_t       *options,
                  const cairo_font_options_t *other);
    ///
     cairo_bool_t
    cairo_font_options_equal (const cairo_font_options_t *options,
                  const cairo_font_options_t *other);
    ///
     ulong
    cairo_font_options_hash (const cairo_font_options_t *options);
    ///
     void
    cairo_font_options_set_antialias (cairo_font_options_t *options,
                      cairo_antialias_t     antialias);
     ///
     cairo_antialias_t
    cairo_font_options_get_antialias (const cairo_font_options_t *options);

    ///
     void
    cairo_font_options_set_subpixel_order (cairo_font_options_t   *options,
                           cairo_subpixel_order_t  subpixel_order);
    ///
     cairo_subpixel_order_t
    cairo_font_options_get_subpixel_order (const cairo_font_options_t *options);
    ///
     void
    cairo_font_options_set_hint_style (cairo_font_options_t *options,
                       cairo_hint_style_t     hint_style);
    ///
     cairo_hint_style_t
    cairo_font_options_get_hint_style (const cairo_font_options_t *options);
    ///
     void
    cairo_font_options_set_hint_metrics (cairo_font_options_t *options,
                         cairo_hint_metrics_t  hint_metrics);
    ///
     cairo_hint_metrics_t
    cairo_font_options_get_hint_metrics (const cairo_font_options_t *options);

    /* This interface is for dealing with text as text, not caring about the
       font object inside the the cairo_t. */
    ///
     void
    cairo_select_font_face (cairo_t              *cr,
                const char           *family,
                cairo_font_slant_t   slant,
                cairo_font_weight_t  weight);
    ///
     void
    cairo_set_font_size (cairo_t* cr, double size);
    ///
     void
    cairo_set_font_matrix (cairo_t		    *cr,
                   const cairo_matrix_t *matrix);
    ///
     void
    cairo_get_font_matrix (cairo_t* cr,
                   cairo_matrix_t *matrix);
    ///
     void
    cairo_set_font_options (cairo_t                    *cr,
                const cairo_font_options_t *options);
    ///
     void
    cairo_get_font_options (cairo_t              *cr,
                cairo_font_options_t *options);
    ///
     void
    cairo_set_font_face (cairo_t* cr, cairo_font_face_t *font_face);
    ///
     cairo_font_face_t *
    cairo_get_font_face (cairo_t* cr);
    ///
     void
    cairo_set_scaled_font (cairo_t                   *cr,
                   const cairo_scaled_font_t *scaled_font);
    ///
     cairo_scaled_font_t *
    cairo_get_scaled_font (cairo_t* cr);
    ///
     void
    cairo_show_text (cairo_t* cr, const char *utf8);
    ///
     void
    cairo_show_glyphs (cairo_t* cr, const cairo_glyph_t *glyphs, int num_glyphs);
    ///
     void
    cairo_show_text_glyphs (cairo_t			   *cr,
                const char		   *utf8,
                int			    utf8_len,
                const cairo_glyph_t	   *glyphs,
                int			    num_glyphs,
                const cairo_text_cluster_t *clusters,
                int			    num_clusters,
                cairo_text_cluster_flags_t  cluster_flags);
    ///
     void
    cairo_text_path  (cairo_t* cr, const char *utf8);
    ///
     void
    cairo_glyph_path (cairo_t* cr, const cairo_glyph_t *glyphs, int num_glyphs);
    ///
     void
    cairo_text_extents (cairo_t              *cr,
                const char    	 *utf8,
                cairo_text_extents_t *extents);
    ///
     void
    cairo_glyph_extents (cairo_t               *cr,
                 const cairo_glyph_t   *glyphs,
                 int                   num_glyphs,
                 cairo_text_extents_t  *extents);
    ///
     void
    cairo_font_extents (cairo_t              *cr,
                cairo_font_extents_t *extents);

    /* Generic identifier for a font style */
    ///
     cairo_font_face_t *
    cairo_font_face_reference (cairo_font_face_t *font_face);
    ///
     void
    cairo_font_face_destroy (cairo_font_face_t *font_face);
    ///
     uint
    cairo_font_face_get_reference_count (cairo_font_face_t *font_face);
    ///
     cairo_status_t
    cairo_font_face_status (cairo_font_face_t *font_face);


    /**
     * $(D cairo_font_type_t) is used to describe the type of a given font
     * face or scaled font. The font types are also known as "font
     * backends" within cairo.
     *
     * The type of a font face is determined by the function used to
     * create it, which will generally be of the form
     * cairo_$(B type)_font_face_create(). The font face type can be queried
     * with cairo_font_face_get_type()
     *
     * The various $(D cairo_font_face_t) functions can be used with a font face
     * of any type.
     *
     * The type of a scaled font is determined by the type of the font
     * face passed to cairo_scaled_font_create(). The scaled font type can
     * be queried with cairo_scaled_font_get_type()
     *
     * The various $(D cairo_scaled_font_t functions) can be used with scaled
     * fonts of any type, but some font backends also provide
     * type-specific functions that must only be called with a scaled font
     * of the appropriate type. These functions have names that begin with
     * cairo_$(B type)_scaled_font() such as cairo_ft_scaled_font_lock_face().
     *
     * The behavior of calling a type-specific function with a scaled font
     * of the wrong type is undefined.
     *
     * New entries may be added in future versions.
     *
     * Since: 1.2
     **/
    enum cairo_font_type_t
    {
        CAIRO_FONT_TYPE_TOY, ///The font was created using cairo's toy font api
        CAIRO_FONT_TYPE_FT, ///The font is of type FreeType
        CAIRO_FONT_TYPE_WIN32, ///The font is of type Win32
        CAIRO_FONT_TYPE_QUARTZ, ///The font is of type Quartz (Since: 1.6)
        CAIRO_FONT_TYPE_USER ///The font was create using cairo's user font api (Since: 1.8)
    }
    ///
     cairo_font_type_t
    cairo_font_face_get_type (cairo_font_face_t *font_face);
    ///
     void *
    cairo_font_face_get_user_data (cairo_font_face_t	   *font_face,
                       const cairo_user_data_key_t *key);
    ///
     cairo_status_t
    cairo_font_face_set_user_data (cairo_font_face_t	   *font_face,
                       const cairo_user_data_key_t *key,
                       void			   *user_data,
                       cairo_destroy_func_t	    destroy);

    /** Portable interface to general font features. */

     cairo_scaled_font_t *
    cairo_scaled_font_create (cairo_font_face_t          *font_face,
                  const cairo_matrix_t       *font_matrix,
                  const cairo_matrix_t       *ctm,
                  const cairo_font_options_t *options);
    ///ditto
     cairo_scaled_font_t *
    cairo_scaled_font_reference (cairo_scaled_font_t *scaled_font);
    ///ditto
     void
    cairo_scaled_font_destroy (cairo_scaled_font_t *scaled_font);
    ///ditto
     uint
    cairo_scaled_font_get_reference_count (cairo_scaled_font_t *scaled_font);
    ///ditto
     cairo_status_t
    cairo_scaled_font_status (cairo_scaled_font_t *scaled_font);
    ///ditto
     cairo_font_type_t
    cairo_scaled_font_get_type (cairo_scaled_font_t *scaled_font);
    ///ditto
     void *
    cairo_scaled_font_get_user_data (cairo_scaled_font_t         *scaled_font,
                     const cairo_user_data_key_t *key);
    ///ditto
     cairo_status_t
    cairo_scaled_font_set_user_data (cairo_scaled_font_t         *scaled_font,
                     const cairo_user_data_key_t *key,
                     void                        *user_data,
                     cairo_destroy_func_t	      destroy);
    ///ditto
     void
    cairo_scaled_font_extents (cairo_scaled_font_t  *scaled_font,
                   cairo_font_extents_t *extents);
    ///ditto
     void
    cairo_scaled_font_text_extents (cairo_scaled_font_t  *scaled_font,
                    const char  	     *utf8,
                    cairo_text_extents_t *extents);
    ///ditto
     void
    cairo_scaled_font_glyph_extents (cairo_scaled_font_t   *scaled_font,
                     const cairo_glyph_t   *glyphs,
                     int                   num_glyphs,
                     cairo_text_extents_t  *extents);
    ///ditto
     cairo_status_t
    cairo_scaled_font_text_to_glyphs (cairo_scaled_font_t        *scaled_font,
                      double		      x,
                      double		      y,
                      const char	             *utf8,
                      int		              utf8_len,
                      cairo_glyph_t	            **glyphs,
                      int		             *num_glyphs,
                      cairo_text_cluster_t      **clusters,
                      int		             *num_clusters,
                      cairo_text_cluster_flags_t *cluster_flags);
    ///ditto
     cairo_font_face_t *
    cairo_scaled_font_get_font_face (cairo_scaled_font_t *scaled_font);
    ///ditto
     void
    cairo_scaled_font_get_font_matrix (cairo_scaled_font_t	*scaled_font,
                       cairo_matrix_t	*font_matrix);
    ///ditto
     void
    cairo_scaled_font_get_ctm (cairo_scaled_font_t	*scaled_font,
                   cairo_matrix_t	*ctm);
    ///ditto
     void
    cairo_scaled_font_get_scale_matrix (cairo_scaled_font_t	*scaled_font,
                        cairo_matrix_t	*scale_matrix);
    ///ditto
     void
    cairo_scaled_font_get_font_options (cairo_scaled_font_t		*scaled_font,
                        cairo_font_options_t	*options);


    /** Toy fonts */

     cairo_font_face_t *
    cairo_toy_font_face_create (const char           *family,
                    cairo_font_slant_t    slant,
                    cairo_font_weight_t   weight);
    ///ditto
     const(char)*
    cairo_toy_font_face_get_family (cairo_font_face_t *font_face);
    ///ditto
     cairo_font_slant_t
    cairo_toy_font_face_get_slant (cairo_font_face_t *font_face);
    ///ditto
     cairo_font_weight_t
    cairo_toy_font_face_get_weight (cairo_font_face_t *font_face);


    /** User fonts */

     cairo_font_face_t *
    cairo_user_font_face_create ();

    /* User-font method signatures */

    /**
     * $(D cairo_user_scaled_font_init_func_t) is the type of function which is
     * called when a scaled-font needs to be created for a user font-face.
     *
     * The cairo context $(D cr) is not used by the caller, but is prepared in font
     * space, similar to what the cairo contexts passed to the render_glyph
     * method will look like.  The callback can use this context for extents
     * computation for example.  After the callback is called, $(D cr) is checked
     * for any error status.
     *
     * The $(D extents) argument is where the user font sets the font extents for
     * $(D scaled_font).  It is in font space, which means that for most cases its
     * ascent and descent members should add to 1.0.  $(D extents) is preset to
     * hold a value of 1.0 for ascent, height, and max_x_advance, and 0.0 for
     * descent and max_y_advance members.
     *
     * The callback is optional.  If not set, default font extents as described
     * in the previous paragraph will be used.
     *
     * Note that $(D scaled_font) is not fully initialized at this
     * point and trying to use it for text operations in the callback will result
     * in deadlock.
     *
     * Params:
     * scaled_font = the scaled-font being created
     * cr = a cairo context, in font space
     * extents = font extents to fill in, in font space
     *
     * Returns: $(D CAIRO_STATUS_SUCCESS) upon success, or an error status on error.
     *
     * Since: 1.8
     **/
    alias extern(C) cairo_status_t function(cairo_scaled_font_t  *scaled_font,
                                      cairo_t              *cr,
                                      cairo_font_extents_t *extents) cairo_user_scaled_font_init_func_t;

    /**
     *
     * $(D cairo_user_scaled_font_render_glyph_func_t) is the type of function which
     * is called when a user scaled-font needs to render a glyph.
     *
     * The callback is mandatory, and expected to draw the glyph with code $(D glyph) to
     * the cairo context $(D cr).  $(D cr) is prepared such that the glyph drawing is done in
     * font space.  That is, the matrix set on $(D cr) is the scale matrix of $(D scaled_font),
     * The $(D extents) argument is where the user font sets the font extents for
     * $(D scaled_font).  However, if user prefers to draw in user space, they can
     * achieve that by changing the matrix on $(D cr).  All cairo rendering operations
     * to $(D cr) are permitted, however, the result is undefined if any source other
     * than the default source on $(D cr) is used.  That means, glyph bitmaps should
     * be rendered using cairo_mask() instead of cairo_paint().
     *
     * Other non-default settings on $(D cr) include a font size of 1.0 (given that
     * it is set up to be in font space), and font options corresponding to
     * $(D scaled_font).
     *
     * The $(D extents) argument is preset to have $(D x_bearing),
     * $(D width), and $(D y_advance) of zero,
     * $(D y_bearing) set to $(D -font_extents.ascent),
     * $(D height) to $(D font_extents.ascent+font_extents.descent),
     * and $(D x_advance) to $(D font_extents.max_x_advance).
     * The only field user needs to set in majority of cases is
     * $(D x_advance).
     * If the $(D width) field is zero upon the callback returning
     * (which is its preset value), the glyph extents are automatically computed
     * based on the drawings done to $(D cr).  This is in most cases exactly what the
     * desired behavior is.  However, if for any reason the callback sets the
     * extents, it must be ink extents, and include the extents of all drawing
     * done to $(D cr) in the callback.
     *
     * Params:
     * scaled_font = user scaled-font
     * glyph = glyph code to render
     * cr = cairo context to draw to, in font space
     * extents = glyph extents to fill in, in font space
     *
     * Returns: $(D CAIRO_STATUS_SUCCESS) upon success, or
     * $(D CAIRO_STATUS_USER_FONT_ERROR) or any other error status on error.
     *
     * Since: 1.8
     **/
    alias extern(C) cairo_status_t function(cairo_scaled_font_t  *scaled_font,
                                          ulong         glyph,
                                          cairo_t              *cr,
                                          cairo_text_extents_t *extents) cairo_user_scaled_font_render_glyph_func_t;

    /**
     * $(D cairo_user_scaled_font_text_to_glyphs_func_t) is the type of function which
     * is called to convert input text to an array of glyphs.  This is used by the
     * cairo_show_text() operation.
     *
     * Using this callback the user-font has full control on glyphs and their
     * positions.  That means, it allows for features like ligatures and kerning,
     * as well as complex $(I shaping) required for scripts like
     * Arabic and Indic.
     *
     * The $(D num_glyphs) argument is preset to the number of glyph entries available
     * in the $(D glyphs) buffer. If the $(D glyphs) buffer is $(D NULL), the value of
     * $(D num_glyphs) will be zero.  If the provided glyph array is too short for
     * the conversion (or for convenience), a new glyph array may be allocated
     * using cairo_glyph_allocate() and placed in $(D glyphs).  Upon return,
     * $(D num_glyphs) should contain the number of generated glyphs.  If the value
     * $(D glyphs) points at has changed after the call, the caller will free the
     * allocated glyph array using cairo_glyph_free().
     * The callback should populate the glyph indices and positions (in font space)
     * assuming that the text is to be shown at the origin.
     *
     * If $(D clusters) is not $(D NULL), $(D num_clusters) and $(D cluster_flags) are also
     * non-$(D NULL), and cluster mapping should be computed. The semantics of how
     * cluster array allocation works is similar to the glyph array.  That is,
     * if $(D clusters) initially points to a non-$(D NULL) value, that array may be used
     * as a cluster buffer, and $(D num_clusters) points to the number of cluster
     * entries available there.  If the provided cluster array is too short for
     * the conversion (or for convenience), a new cluster array may be allocated
     * using cairo_text_cluster_allocate() and placed in $(D clusters).  Upon return,
     * $(D num_clusters) should contain the number of generated clusters.
     * If the value $(D clusters) points at has changed after the call, the caller
     * will free the allocated cluster array using cairo_text_cluster_free().
     *
     * The callback is optional.  If $(D num_glyphs) is negative upon
     * the callback returning or if the return value
     * is $(D CAIRO_STATUS_USER_FONT_NOT_IMPLEMENTED), the unicode_to_glyph callback
     * is tried.  See $(D cairo_user_scaled_font_unicode_to_glyph_func_t).
     *
     * Note: While cairo does not impose any limitation on glyph indices,
     * some applications may assume that a glyph index fits in a 16-bit
     * unsigned integer.  As such, it is advised that user-fonts keep their
     * glyphs in the 0 to 65535 range.  Furthermore, some applications may
     * assume that glyph 0 is a special glyph-not-found glyph.  User-fonts
     * are advised to use glyph 0 for such purposes and do not use that
     * glyph value for other purposes.
     *
     * Params:
     * scaled_font = the scaled-font being created
     * utf8 = a string of text encoded in UTF-8
     * utf8_len = length of @utf8 in bytes
     * glyphs = pointer to array of glyphs to fill, in font space
     * num_glyphs = pointer to number of glyphs
     * clusters = pointer to array of cluster mapping information to fill, or %NULL
     * num_clusters = pointer to number of clusters
     * cluster_flags = pointer to location to store cluster flags corresponding to the
     *                 output @clusters
     *
     * Returns: $(D CAIRO_STATUS_SUCCESS) upon success,
     * $(D CAIRO_STATUS_USER_FONT_NOT_IMPLEMENTED) if fallback options should be tried,
     * or $(D CAIRO_STATUS_USER_FONT_ERROR) or any other error status on error.
     *
     * Since: 1.8
     **/
    alias extern(C) cairo_status_t function(cairo_scaled_font_t        *scaled_font,
                                        const char	           *utf8,
                                        int		            utf8_len,
                                        cairo_glyph_t	          **glyphs,
                                        int		           *num_glyphs,
                                        cairo_text_cluster_t      **clusters,
                                        int		           *num_clusters,
                                        cairo_text_cluster_flags_t *cluster_flags) cairo_user_scaled_font_text_to_glyphs_func_t;

    /**
     * $(D cairo_user_scaled_font_unicode_to_glyph_func_t) is the type of function which
     * is called to convert an input Unicode character to a single glyph.
     * This is used by the cairo_show_text() operation.
     *
     * This callback is used to provide the same functionality as the
     * text_to_glyphs callback does (see $(D cairo_user_scaled_font_text_to_glyphs_func_t))
     * but has much less control on the output,
     * in exchange for increased ease of use.  The inherent assumption to using
     * this callback is that each character maps to one glyph, and that the
     * mapping is context independent.  It also assumes that glyphs are positioned
     * according to their advance width.  These mean no ligatures, kerning, or
     * complex scripts can be implemented using this callback.
     *
     * The callback is optional, and only used if text_to_glyphs callback is not
     * set or fails to return glyphs.  If this callback is not set or if it returns
     * $(D CAIRO_STATUS_USER_FONT_NOT_IMPLEMENTED), an identity mapping from Unicode
     * code-points to glyph indices is assumed.
     *
     * Note: While cairo does not impose any limitation on glyph indices,
     * some applications may assume that a glyph index fits in a 16-bit
     * unsigned integer.  As such, it is advised that user-fonts keep their
     * glyphs in the 0 to 65535 range.  Furthermore, some applications may
     * assume that glyph 0 is a special glyph-not-found glyph.  User-fonts
     * are advised to use glyph 0 for such purposes and do not use that
     * glyph value for other purposes.
     *
     * Params:
     * scaled_font = the scaled-font being created
     * unicode = input unicode character code-point
     * glyph_index = output glyph index
     *
     * Returns: $(D CAIRO_STATUS_SUCCESS) upon success,
     * $(D CAIRO_STATUS_USER_FONT_NOT_IMPLEMENTED) if fallback options should be tried,
     * or $(D CAIRO_STATUS_USER_FONT_ERROR) or any other error status on error.
     *
     * Since: 1.8
     **/
    alias extern(C) cairo_status_t function(cairo_scaled_font_t *scaled_font,
                                          ulong        unicode,
                                          ulong       *glyph_index) cairo_user_scaled_font_unicode_to_glyph_func_t;

    /** User-font method setters */

     void
    cairo_user_font_face_set_init_func (cairo_font_face_t                  *font_face,
                        cairo_user_scaled_font_init_func_t  init_func);
    ///ditto
     void
    cairo_user_font_face_set_render_glyph_func (cairo_font_face_t                          *font_face,
                            cairo_user_scaled_font_render_glyph_func_t  render_glyph_func);
    ///ditto
     void
    cairo_user_font_face_set_text_to_glyphs_func (cairo_font_face_t                            *font_face,
                              cairo_user_scaled_font_text_to_glyphs_func_t  text_to_glyphs_func);
    ///ditto
     void
    cairo_user_font_face_set_unicode_to_glyph_func (cairo_font_face_t                              *font_face,
                                cairo_user_scaled_font_unicode_to_glyph_func_t  unicode_to_glyph_func);

    /** User-font method getters */

     cairo_user_scaled_font_init_func_t
    cairo_user_font_face_get_init_func (cairo_font_face_t *font_face);
    ///ditto
     cairo_user_scaled_font_render_glyph_func_t
    cairo_user_font_face_get_render_glyph_func (cairo_font_face_t *font_face);
    ///ditto
     cairo_user_scaled_font_text_to_glyphs_func_t
    cairo_user_font_face_get_text_to_glyphs_func (cairo_font_face_t *font_face);
    ///ditto
     cairo_user_scaled_font_unicode_to_glyph_func_t
    cairo_user_font_face_get_unicode_to_glyph_func (cairo_font_face_t *font_face);


    /** Query functions */

     cairo_operator_t
    cairo_get_operator (cairo_t* cr);
    ///ditto
     cairo_pattern_t *
    cairo_get_source (cairo_t* cr);
    ///ditto
     double
    cairo_get_tolerance (cairo_t* cr);
    ///ditto
     cairo_antialias_t
    cairo_get_antialias (cairo_t* cr);
    ///ditto
     cairo_bool_t
    cairo_has_current_point (cairo_t* cr);
    ///ditto
     void
    cairo_get_current_point (cairo_t* cr, double *x, double *y);
    ///ditto
     cairo_fill_rule_t
    cairo_get_fill_rule (cairo_t* cr);
    ///ditto
     double
    cairo_get_line_width (cairo_t* cr);
    ///ditto
     cairo_line_cap_t
    cairo_get_line_cap (cairo_t* cr);
    ///ditto
     cairo_line_join_t
    cairo_get_line_join (cairo_t* cr);
    ///ditto
     double
    cairo_get_miter_limit (cairo_t* cr);
    ///ditto
     int
    cairo_get_dash_count (cairo_t* cr);
    ///ditto
     void
    cairo_get_dash (cairo_t* cr, double *dashes, double *offset);
    ///ditto
     void
    cairo_get_matrix (cairo_t* cr, cairo_matrix_t *matrix);
    ///ditto
     cairo_surface_t *
    cairo_get_target (cairo_t* cr);
    ///ditto
     cairo_surface_t *
    cairo_get_group_target (cairo_t* cr);

    /**
     * $(D cairo_path_data_t) is used to describe the type of one portion
     * of a path when represented as a $(D cairo_path_t).
     * See $(D cairo_path_data_t) for details.
     **/
    enum cairo_path_data_type_t
    {
        CAIRO_PATH_MOVE_TO, ///A move-to operation
        CAIRO_PATH_LINE_TO, ///A line-to operation
        CAIRO_PATH_CURVE_TO, ///A curve-to operation
        CAIRO_PATH_CLOSE_PATH ///A close-path operation
    }

    /**
     * $(D cairo_path_data_t) is used to represent the path data inside a
     * $(D cairo_path_t).
     *
     * The data structure is designed to try to balance the demands of
     * efficiency and ease-of-use. A path is represented as an array of
     * $(D cairo_path_data_t), which is a union of headers and points.
     *
     * Each portion of the path is represented by one or more elements in
     * the array, (one header followed by 0 or more points). The length
     * value of the header is the number of array elements for the current
     * portion including the header, (ie. length == 1 + # of points), and
     * where the number of points for each element type is as follows:
     *
     * --------------------
     *     $(D CAIRO_PATH_MOVE_TO):     1 point
     *     $(D CAIRO_PATH_LINE_TO):     1 point
     *     $(D CAIRO_PATH_CURVE_TO):    3 points
     *     $(D CAIRO_PATH_CLOSE_PATH):  0 points
     * --------------------
     *
     * The semantics and ordering of the coordinate values are consistent
     * with cairo_move_to(), cairo_line_to(), cairo_curve_to(), and
     * cairo_close_path().
     *
     * Examples:
     * Here is sample code for iterating through a $(D cairo_path_t):
     *
     * --------------------
     *      int i;
     *      cairo_path_t *path;
     *      cairo_path_data_t *data;
     *
     *      path = cairo_copy_path (cr);
     *
     *      for (i=0; i < path->num_data; i += path->data[i].header.length) {
     *          data = &amp;path->data[i];
     *          switch (data->header.type) {
     *          case CAIRO_PATH_MOVE_TO:
     *              do_move_to_things (data[1].point.x, data[1].point.y);
     *              break;
     *          case CAIRO_PATH_LINE_TO:
     *              do_line_to_things (data[1].point.x, data[1].point.y);
     *              break;
     *          case CAIRO_PATH_CURVE_TO:
     *              do_curve_to_things (data[1].point.x, data[1].point.y,
     *                                  data[2].point.x, data[2].point.y,
     *                                  data[3].point.x, data[3].point.y);
     *              break;
     *          case CAIRO_PATH_CLOSE_PATH:
     *              do_close_path_things ();
     *              break;
     *          }
     *      }
     *      cairo_path_destroy (path);
     * --------------------
     *
     * As of cairo 1.4, cairo does not mind if there are more elements in
     * a portion of the path than needed.  Such elements can be used by
     * users of the cairo API to hold extra values in the path data
     * structure.  For this reason, it is recommended that applications
     * always use $(D data->header.length) to
     * iterate over the path data, instead of hardcoding the number of
     * elements for each element type.
     **/
    struct cairo_path_data_t
    {
        union
        {
            PathDataHeader header; ///
            PathDataPoint point; ///
        }
    }
    ///
    struct PathDataHeader
    {
        cairo_path_data_type_t type;///
        int length;///
    }
    ///
    struct PathDataPoint
    {
        double x, y; ///
    }

    /**
     * A data structure for holding a path. This data structure serves as
     * the return value for cairo_copy_path() and
     * cairo_copy_path_flat() as well the input value for
     * cairo_append_path().
     *
     * See $(D cairo_path_data_t) for hints on how to iterate over the
     * actual data within the path.
     *
     * The num_data member gives the number of elements in the data
     * array. This number is larger than the number of independent path
     * portions (defined in $(D cairo_path_data_type_t)), since the data
     * includes both headers and coordinates for each portion.
     **/
    struct cairo_path_t
    {
        cairo_status_t status; ///the current error status
        cairo_path_data_t *data; ///the elements in the path
        int num_data; ///the number of elements in the data array
    }
    ///
     cairo_path_t *
    cairo_copy_path (cairo_t* cr);
    ///
     cairo_path_t *
    cairo_copy_path_flat (cairo_t* cr);
    ///
     void
    cairo_append_path (cairo_t		*cr,
               const cairo_path_t	*path);
    ///
     void
    cairo_path_destroy (cairo_path_t *path);

    /** Error status queries */

     cairo_status_t
    cairo_status (cairo_t* cr);
    ///ditto
     immutable(char)*
    cairo_status_to_string (cairo_status_t status);

    /** Backend device manipulation */

     cairo_device_t *
    cairo_device_reference (cairo_device_t *device);

    /**
     * $(D cairo_device_type_t) is used to describe the type of a given
     * device. The devices types are also known as "backends" within cairo.
     *
     * The device type can be queried with cairo_device_get_type()
     *
     * The various $(D cairo_device_t) functions can be used with surfaces of
     * any type, but some backends also provide type-specific functions
     * that must only be called with a device of the appropriate
     * type. These functions have names that begin with
     * cairo_$(B type)_device<...> such as cairo_xcb_device_debug_set_render_version().
     *
     * The behavior of calling a type-specific function with a surface of
     * the wrong type is undefined.
     *
     * New entries may be added in future versions.
     *
     * Since: 1.10
     **/
    enum cairo_device_type_t
    {
        CAIRO_DEVICE_TYPE_DRM, ///The surface is of type Direct Render Manager
        CAIRO_DEVICE_TYPE_GL, ///The surface is of type OpenGL
        CAIRO_DEVICE_TYPE_SCRIPT, ///The surface is of type script
        CAIRO_DEVICE_TYPE_XCB, ///The surface is of type xcb
        CAIRO_DEVICE_TYPE_XLIB, ///The surface is of type xlib
        CAIRO_DEVICE_TYPE_XML, ///The surface is of type XML
    }
    ///
     cairo_device_type_t
    cairo_device_get_type (cairo_device_t *device);
    ///
     cairo_status_t
    cairo_device_status (cairo_device_t *device);
    ///
     cairo_status_t
    cairo_device_acquire (cairo_device_t *device);
    ///
     void
    cairo_device_release (cairo_device_t *device);
    ///
     void
    cairo_device_flush (cairo_device_t *device);
    ///
     void
    cairo_device_finish (cairo_device_t *device);
    ///
     void
    cairo_device_destroy (cairo_device_t *device);
    ///
     uint
    cairo_device_get_reference_count (cairo_device_t *device);
    ///
     void *
    cairo_device_get_user_data (cairo_device_t		 *device,
                    const cairo_user_data_key_t *key);
    ///
     cairo_status_t
    cairo_device_set_user_data (cairo_device_t		 *device,
                    const cairo_user_data_key_t *key,
                    void			 *user_data,
                    cairo_destroy_func_t	  destroy);


    /** Surface manipulation */

     cairo_surface_t *
    cairo_surface_create_similar (cairo_surface_t  *other,
                      cairo_content_t	content,
                      int		width,
                      int		height);
    ///ditto
     cairo_surface_t *
    cairo_surface_create_for_rectangle (cairo_surface_t	*target,
                                        double		 x,
                                        double		 y,
                                        double		 width,
                                        double		 height);
    ///ditto
     cairo_surface_t *
    cairo_surface_reference (cairo_surface_t *surface);
    ///ditto
     void
    cairo_surface_finish (cairo_surface_t *surface);
    ///ditto
     void
    cairo_surface_destroy (cairo_surface_t *surface);
    ///ditto
     cairo_device_t *
    cairo_surface_get_device (cairo_surface_t *surface);
    ///ditto
     uint
    cairo_surface_get_reference_count (cairo_surface_t *surface);
    ///ditto
     cairo_status_t
    cairo_surface_status (cairo_surface_t *surface);

    /**
     * $(D cairo_surface_type_t) is used to describe the type of a given
     * surface. The surface types are also known as "backends" or "surface
     * backends" within cairo.
     *
     * The type of a surface is determined by the function used to create
     * it, which will generally be of the form cairo_$(B type)_surface_create(),
     * (though see cairo_surface_create_similar() as well).
     *
     * The surface type can be queried with cairo_surface_get_type()
     *
     * The various $(D cairo_surface_t) functions can be used with surfaces of
     * any type, but some backends also provide type-specific functions
     * that must only be called with a surface of the appropriate
     * type. These functions have names that begin with
     * cairo_$(B type)_surface<...> such as cairo_image_surface_get_width().
     *
     * The behavior of calling a type-specific function with a surface of
     * the wrong type is undefined.
     *
     * New entries may be added in future versions.
     *
     * Since: 1.2
     **/
    enum cairo_surface_type_t
    {
        CAIRO_SURFACE_TYPE_IMAGE, ///The surface is of type image
        CAIRO_SURFACE_TYPE_PDF, ///The surface is of type pdf
        CAIRO_SURFACE_TYPE_PS, ///The surface is of type ps
        CAIRO_SURFACE_TYPE_XLIB, ///The surface is of type xlib
        CAIRO_SURFACE_TYPE_XCB, ///The surface is of type xcb
        CAIRO_SURFACE_TYPE_GLITZ, ///The surface is of type glitz
        CAIRO_SURFACE_TYPE_QUARTZ, ///The surface is of type quartz
        CAIRO_SURFACE_TYPE_WIN32, ///The surface is of type win32
        CAIRO_SURFACE_TYPE_BEOS, ///The surface is of type beos
        CAIRO_SURFACE_TYPE_DIRECTFB, ///The surface is of type directfb
        CAIRO_SURFACE_TYPE_SVG, ///The surface is of type svg
        CAIRO_SURFACE_TYPE_OS2, ///The surface is of type os2
        CAIRO_SURFACE_TYPE_WIN32_PRINTING, ///The surface is a win32 printing surface
        CAIRO_SURFACE_TYPE_QUARTZ_IMAGE, ///The surface is of type quartz_image
        CAIRO_SURFACE_TYPE_SCRIPT, ///The surface is of type script, since 1.10
        CAIRO_SURFACE_TYPE_QT, ///The surface is of type Qt, since 1.10
        CAIRO_SURFACE_TYPE_RECORDING, ///The surface is of type recording, since 1.10
        CAIRO_SURFACE_TYPE_VG, ///The surface is a OpenVG surface, since 1.10
        CAIRO_SURFACE_TYPE_GL, ///The surface is of type OpenGL, since 1.10
        CAIRO_SURFACE_TYPE_DRM, ///The surface is of type Direct Render Manager, since 1.10
        CAIRO_SURFACE_TYPE_TEE, ///The surface is of type 'tee' (a multiplexing surface), since 1.10
        CAIRO_SURFACE_TYPE_XML, ///The surface is of type XML (for debugging), since 1.10
        CAIRO_SURFACE_TYPE_SKIA, ///The surface is of type Skia, since 1.10
        /**
         * The surface is a subsurface created with
         * cairo_surface_create_for_rectangle(), since 1.10
         */
        CAIRO_SURFACE_TYPE_SUBSURFACE
    }
    ///
     cairo_surface_type_t
    cairo_surface_get_type (cairo_surface_t *surface);
    ///
     cairo_content_t
    cairo_surface_get_content (cairo_surface_t *surface);

    static if(CAIRO_HAS_PNG_FUNCTIONS)
    {
    ///requires -version=CAIRO_HAS_PNG_FUNCTIONS
     cairo_status_t
    cairo_surface_write_to_png (cairo_surface_t	*surface,
                    const char		*filename);
    ///ditto
     cairo_status_t
    cairo_surface_write_to_png_stream (cairo_surface_t	*surface,
                       cairo_write_func_t	write_func,
                       void			*closure);

    }
    ///
     void *
    cairo_surface_get_user_data (cairo_surface_t		 *surface,
                     const cairo_user_data_key_t *key);
    ///
     cairo_status_t
    cairo_surface_set_user_data (cairo_surface_t		 *surface,
                     const cairo_user_data_key_t *key,
                     void			 *user_data,
                     cairo_destroy_func_t	 destroy);
    ///
    enum string CAIRO_MIME_TYPE_JPEG = "image/jpeg";
    ///
    enum string CAIRO_MIME_TYPE_PNG = "image/png";
    ///
    enum string CAIRO_MIME_TYPE_JP2 = "image/jp2";
    ///
    enum string CAIRO_MIME_TYPE_URI = "text/x-uri";
    ///
     void
    cairo_surface_get_mime_data (cairo_surface_t		*surface,
                                 const char			*mime_type,
                                 const ubyte       **data,
                                 ulong		*length);
    ///
     cairo_status_t
    cairo_surface_set_mime_data (cairo_surface_t		*surface,
                                 const char			*mime_type,
                                 const ubyte	*data,
                                 ulong		 length,
                     cairo_destroy_func_t	 destroy,
                     void			*closure);
    ///
     void
    cairo_surface_get_font_options (cairo_surface_t      *surface,
                    cairo_font_options_t *options);
    ///
     void
    cairo_surface_flush (cairo_surface_t *surface);
    ///
     void
    cairo_surface_mark_dirty (cairo_surface_t *surface);
    ///
     void
    cairo_surface_mark_dirty_rectangle (cairo_surface_t *surface,
                        int              x,
                        int              y,
                        int              width,
                        int              height);
    ///
     void
    cairo_surface_set_device_offset (cairo_surface_t *surface,
                     double           x_offset,
                     double           y_offset);
    ///
     void
    cairo_surface_get_device_offset (cairo_surface_t *surface,
                     double          *x_offset,
                     double          *y_offset);
    ///
     void
    cairo_surface_set_fallback_resolution (cairo_surface_t	*surface,
                           double		 x_pixels_per_inch,
                           double		 y_pixels_per_inch);
    ///
     void
    cairo_surface_get_fallback_resolution (cairo_surface_t	*surface,
                           double		*x_pixels_per_inch,
                           double		*y_pixels_per_inch);
    ///
     void
    cairo_surface_copy_page (cairo_surface_t *surface);
    ///
     void
    cairo_surface_show_page (cairo_surface_t *surface);
    ///
     cairo_bool_t
    cairo_surface_has_show_text_glyphs (cairo_surface_t *surface);

    /* Image-surface functions */

    /**
     * $(D cairo_format_t) is used to identify the memory format of
     * image data.
     *
     * New entries may be added in future versions.
     **/
    enum cairo_format_t
    {
        CAIRO_FORMAT_INVALID   = -1, ///no such format exists or is supported.
        /**
         * each pixel is a 32-bit quantity, with
         * alpha in the upper 8 bits, then red, then green, then blue.
         * The 32-bit quantities are stored native-endian. Pre-multiplied
         * alpha is used. (That is, 50% transparent red is 0x80800000,
         * not 0x80ff0000.)
         */
        CAIRO_FORMAT_ARGB32    = 0,
        /**
         * each pixel is a 32-bit quantity, with
         * the upper 8 bits unused. Red, Green, and Blue are stored
         * in the remaining 24 bits in that order.
         */
        CAIRO_FORMAT_RGB24     = 1,
        /**
         * each pixel is a 8-bit quantity holding
         * an alpha value.
         */
        CAIRO_FORMAT_A8        = 2,
        /**
         * each pixel is a 1-bit quantity holding
         * an alpha value. Pixels are packed together into 32-bit
         * quantities. The ordering of the bits matches the
         * endianess of the platform. On a big-endian machine, the
         * first pixel is in the uppermost bit, on a little-endian
         * machine the first pixel is in the least-significant bit.
         */
        CAIRO_FORMAT_A1        = 3,
        /**
         * each pixel is a 16-bit quantity
         * with red in the upper 5 bits, then green in the middle
         * 6 bits, and blue in the lower 5 bits.
         */
        CAIRO_FORMAT_RGB16_565 = 4
    }
    ///
     cairo_surface_t *
    cairo_image_surface_create (cairo_format_t	format,
                    int			width,
                    int			height);
    ///
     int
    cairo_format_stride_for_width (cairo_format_t	format,
                       int		width);
    ///
     cairo_surface_t *
    cairo_image_surface_create_for_data (ubyte	       *data,
                         cairo_format_t		format,
                         int			width,
                         int			height,
                         int			stride);
    ///
     ubyte *
    cairo_image_surface_get_data (cairo_surface_t *surface);
    ///
     cairo_format_t
    cairo_image_surface_get_format (cairo_surface_t *surface);
    ///
     int
    cairo_image_surface_get_width (cairo_surface_t *surface);
    ///
     int
    cairo_image_surface_get_height (cairo_surface_t *surface);
    ///
     int
    cairo_image_surface_get_stride (cairo_surface_t *surface);

    static if(CAIRO_HAS_PNG_FUNCTIONS)
    {
    ///requires -version=CAIRO_HAS_PNG_FUNCTIONS
     cairo_surface_t *
    cairo_image_surface_create_from_png (const char	*filename);
    ///ditto
     cairo_surface_t *
    cairo_image_surface_create_from_png_stream (cairo_read_func_t	read_func,
                            void		*closure);

    }

    /** Recording-surface functions */

     cairo_surface_t *
    cairo_recording_surface_create (cairo_content_t		 content,
                                    const cairo_rectangle_t *extents);
    ///ditto
     void
    cairo_recording_surface_ink_extents (cairo_surface_t *surface,
                                         double *x0,
                                         double *y0,
                                         double *width,
                                         double *height);

    /** Pattern creation functions */

     cairo_pattern_t *
    cairo_pattern_create_rgb (double red, double green, double blue);
    ///ditto
     cairo_pattern_t *
    cairo_pattern_create_rgba (double red, double green, double blue,
                   double alpha);
    ///ditto
     cairo_pattern_t *
    cairo_pattern_create_for_surface (cairo_surface_t *surface);
    ///ditto
     cairo_pattern_t *
    cairo_pattern_create_linear (double x0, double y0,
                     double x1, double y1);
    ///ditto
     cairo_pattern_t *
    cairo_pattern_create_radial (double cx0, double cy0, double radius0,
                     double cx1, double cy1, double radius1);
    ///ditto
     cairo_pattern_t *
    cairo_pattern_reference (cairo_pattern_t *pattern);
    ///ditto
     void
    cairo_pattern_destroy (cairo_pattern_t *pattern);
    ///ditto
     uint
    cairo_pattern_get_reference_count (cairo_pattern_t *pattern);
    ///ditto
     cairo_status_t
    cairo_pattern_status (cairo_pattern_t *pattern);
    ///ditto
     void *
    cairo_pattern_get_user_data (cairo_pattern_t		 *pattern,
                     const cairo_user_data_key_t *key);
    ///ditto
     cairo_status_t
    cairo_pattern_set_user_data (cairo_pattern_t		 *pattern,
                     const cairo_user_data_key_t *key,
                     void			 *user_data,
                     cairo_destroy_func_t	  destroy);

    /**
     * $(D cairo_pattern_type_t) is used to describe the type of a given pattern.
     *
     * The type of a pattern is determined by the function used to create
     * it. The cairo_pattern_create_rgb() and cairo_pattern_create_rgba()
     * functions create SOLID patterns. The remaining
     * cairo_pattern_create<...> functions map to pattern types in obvious
     * ways.
     *
     * The pattern type can be queried with cairo_pattern_get_type()
     *
     * Most $(D cairo_pattern_t) functions can be called with a pattern of any
     * type, (though trying to change the extend or filter for a solid
     * pattern will have no effect). A notable exception is
     * cairo_pattern_add_color_stop_rgb() and
     * cairo_pattern_add_color_stop_rgba() which must only be called with
     * gradient patterns (either LINEAR or RADIAL). Otherwise the pattern
     * will be shutdown and put into an error state.
     *
     * New entries may be added in future versions.
     *
     * Since: 1.2
     **/
    enum cairo_pattern_type_t
    {
        /**
         * The pattern is a solid (uniform)
         * color. It may be opaque or translucent.
         */
        CAIRO_PATTERN_TYPE_SOLID,
        CAIRO_PATTERN_TYPE_SURFACE, ///The pattern is a based on a surface (an image).
        CAIRO_PATTERN_TYPE_LINEAR, ///The pattern is a linear gradient.
        CAIRO_PATTERN_TYPE_RADIAL ///The pattern is a radial gradient.
    }
    ///
     cairo_pattern_type_t
    cairo_pattern_get_type (cairo_pattern_t *pattern);
    ///
     void
    cairo_pattern_add_color_stop_rgb (cairo_pattern_t *pattern,
                      double offset,
                      double red, double green, double blue);
    ///
     void
    cairo_pattern_add_color_stop_rgba (cairo_pattern_t *pattern,
                       double offset,
                       double red, double green, double blue,
                       double alpha);
    ///
     void
    cairo_pattern_set_matrix (cairo_pattern_t      *pattern,
                  const cairo_matrix_t *matrix);
    ///
     void
    cairo_pattern_get_matrix (cairo_pattern_t *pattern,
                  cairo_matrix_t  *matrix);

    /**
     * $(D cairo_extend_t) is used to describe how pattern color/alpha will be
     * determined for areas "outside" the pattern's natural area, (for
     * example, outside the surface bounds or outside the gradient
     * geometry).
     *
     * The default extend mode is $(D CAIRO_EXTEND_NONE) for surface patterns
     * and $(D CAIRO_EXTEND_PAD) for gradient patterns.
     *
     * New entries may be added in future versions.
     **/
    enum cairo_extend_t
    {
        /**
         * pixels outside of the source pattern
         * are fully transparent
         */
        CAIRO_EXTEND_NONE,
        CAIRO_EXTEND_REPEAT, ///the pattern is tiled by repeating
        /**
         * the pattern is tiled by reflecting
         * at the edges (Implemented for surface patterns since 1.6)
         */
        CAIRO_EXTEND_REFLECT,
        /**
         * pixels outside of the pattern copy
         * the closest pixel from the source (Since 1.2; but only
         * implemented for surface patterns since 1.6)
         */
        CAIRO_EXTEND_PAD
    }
    ///
     void
    cairo_pattern_set_extend (cairo_pattern_t *pattern, cairo_extend_t extend);
    ///
     cairo_extend_t
    cairo_pattern_get_extend (cairo_pattern_t *pattern);

    /**
     * $(D cairo_filter_t) is used to indicate what filtering should be
     * applied when reading pixel values from patterns. See
     * cairo_pattern_set_source() for indicating the desired filter to be
     * used with a particular pattern.
     */
    enum cairo_filter_t
    {
        /**
         * A high-performance filter, with quality similar
         * to %CAIRO_FILTER_NEAREST
         */
        CAIRO_FILTER_FAST,
        /**
         * A reasonable-performance filter, with quality
         * similar to %CAIRO_FILTER_BILINEAR
         */
        CAIRO_FILTER_GOOD,
        /**
         * The highest-quality available, performance may
         * not be suitable for interactive use.
         */
        CAIRO_FILTER_BEST,
        /**
         * Nearest-neighbor filtering
         */
        CAIRO_FILTER_NEAREST,
        /**
         * Linear interpolation in two dimensions
         */
        CAIRO_FILTER_BILINEAR,
        /**
         * This filter value is currently
         * unimplemented, and should not be used in current code.
         */
        CAIRO_FILTER_GAUSSIAN
    }
    ///
     void
    cairo_pattern_set_filter (cairo_pattern_t *pattern, cairo_filter_t filter);
    ///
     cairo_filter_t
    cairo_pattern_get_filter (cairo_pattern_t *pattern);
    ///
     cairo_status_t
    cairo_pattern_get_rgba (cairo_pattern_t *pattern,
                double *red, double *green,
                double *blue, double *alpha);
    ///
     cairo_status_t
    cairo_pattern_get_surface (cairo_pattern_t *pattern,
                   cairo_surface_t **surface);

    ///
     cairo_status_t
    cairo_pattern_get_color_stop_rgba (cairo_pattern_t *pattern,
                       int index, double *offset,
                       double *red, double *green,
                       double *blue, double *alpha);
    ///
     cairo_status_t
    cairo_pattern_get_color_stop_count (cairo_pattern_t *pattern,
                        int *count);
    ///
     cairo_status_t
    cairo_pattern_get_linear_points (cairo_pattern_t *pattern,
                     double *x0, double *y0,
                     double *x1, double *y1);
    ///
     cairo_status_t
    cairo_pattern_get_radial_circles (cairo_pattern_t *pattern,
                      double *x0, double *y0, double *r0,
                      double *x1, double *y1, double *r1);

    /** Matrix functions */

     void
    cairo_matrix_init (cairo_matrix_t *matrix,
               double  xx, double  yx,
               double  xy, double  yy,
               double  x0, double  y0);
    ///ditto
     void
    cairo_matrix_init_identity (cairo_matrix_t *matrix);
    ///ditto
     void
    cairo_matrix_init_translate (cairo_matrix_t *matrix,
                     double tx, double ty);
    ///ditto
     void
    cairo_matrix_init_scale (cairo_matrix_t *matrix,
                 double sx, double sy);
    ///ditto
     void
    cairo_matrix_init_rotate (cairo_matrix_t *matrix,
                  double radians);
    ///ditto
     void
    cairo_matrix_translate (cairo_matrix_t *matrix, double tx, double ty);
    ///ditto
     void
    cairo_matrix_scale (cairo_matrix_t *matrix, double sx, double sy);
    ///ditto
     void
    cairo_matrix_rotate (cairo_matrix_t *matrix, double radians);
    ///ditto
     cairo_status_t
    cairo_matrix_invert (cairo_matrix_t *matrix);
    ///ditto
     void
    cairo_matrix_multiply (cairo_matrix_t	    *result,
                   const cairo_matrix_t *a,
                   const cairo_matrix_t *b);
    ///ditto
     void
    cairo_matrix_transform_distance (const cairo_matrix_t *matrix,
                     double *dx, double *dy);
    ///ditto
     void
    cairo_matrix_transform_point (const cairo_matrix_t *matrix,
                      double *x, double *y);

    /* Region functions */

    /**
     * A $(D cairo_region_t) represents a set of integer-aligned rectangles.
     *
     * It allows set-theoretical operations like cairo_region_union() and
     * cairo_region_intersect() to be performed on them.
     *
     * Memory management of $(D cairo_region_t) is done with
     * cairo_region_reference() and cairo_region_destroy().
     *
     * Since: 1.10
     **/
    struct cairo_region_t {};

    /**
     * A data structure for holding a rectangle with integer coordinates.
     *
     * Since: 1.10
     **/

    struct cairo_rectangle_int_t
    {
        int x; ///X coordinate of the left side of the rectangle
        int y; ///Y coordinate of the the top side of the rectangle
        int width; ///width of the rectangle
        int height; ///height of the rectangle
    }
    ///
    enum cairo_region_overlap_t
    {
        CAIRO_REGION_OVERLAP_IN,		/** completely inside region */
        CAIRO_REGION_OVERLAP_OUT,		/** completely outside region */
        CAIRO_REGION_OVERLAP_PART		/** partly inside region */
    }
    ///
     cairo_region_t *
    cairo_region_create ();
    ///
     cairo_region_t *
    cairo_region_create_rectangle (const cairo_rectangle_int_t *rectangle);
    ///
     cairo_region_t *
    cairo_region_create_rectangles (const cairo_rectangle_int_t *rects,
                    int count);
    ///
     cairo_region_t *
    cairo_region_copy (const cairo_region_t *original);
    ///
     cairo_region_t *
    cairo_region_reference (cairo_region_t *region);
    ///
     void
    cairo_region_destroy (cairo_region_t *region);
    ///
     cairo_bool_t
    cairo_region_equal (const cairo_region_t *a, const cairo_region_t *b);
    ///
     cairo_status_t
    cairo_region_status (const cairo_region_t *region);
    ///
     void
    cairo_region_get_extents (const cairo_region_t        *region,
                  cairo_rectangle_int_t *extents);
    ///
     int
    cairo_region_num_rectangles (const cairo_region_t *region);
    ///
     void
    cairo_region_get_rectangle (const cairo_region_t  *region,
                    int                    nth,
                    cairo_rectangle_int_t *rectangle);
    ///
     cairo_bool_t
    cairo_region_is_empty (const cairo_region_t *region);
    ///
     cairo_region_overlap_t
    cairo_region_contains_rectangle (const cairo_region_t *region,
                     const cairo_rectangle_int_t *rectangle);
    ///
     cairo_bool_t
    cairo_region_contains_point (const cairo_region_t *region, int x, int y);
    ///
     void
    cairo_region_translate (cairo_region_t *region, int dx, int dy);
    ///
     cairo_status_t
    cairo_region_subtract (cairo_region_t *dst, const cairo_region_t *other);
    ///
     cairo_status_t
    cairo_region_subtract_rectangle (cairo_region_t *dst,
                     const cairo_rectangle_int_t *rectangle);
    ///
     cairo_status_t
    cairo_region_intersect (cairo_region_t *dst, const cairo_region_t *other);
    ///
     cairo_status_t
    cairo_region_intersect_rectangle (cairo_region_t *dst,
                      const cairo_rectangle_int_t *rectangle);
    ///
     cairo_status_t
    cairo_region_union (cairo_region_t *dst, const cairo_region_t *other);
    ///
     cairo_status_t
    cairo_region_union_rectangle (cairo_region_t *dst,
                      const cairo_rectangle_int_t *rectangle);
    ///
     cairo_status_t
    cairo_region_xor (cairo_region_t *dst, const cairo_region_t *other);
    ///
     cairo_status_t
    cairo_region_xor_rectangle (cairo_region_t *dst,
                    const cairo_rectangle_int_t *rectangle);

    /** Functions to be used while debugging (not intended for use in production code) */
     void
    cairo_debug_reset_static_data ();
}
