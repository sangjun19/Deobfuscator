/*=============================================================================
   Copyright (c) 2016-2023 Joel de Guzman

   Distributed under the MIT License [ https://opensource.org/licenses/MIT ]
=============================================================================*/
#include <elements/element/port.hpp>
#include <elements/element/traversal.hpp>
#include <elements/view.hpp>
#include <algorithm>
#include <cmath>

namespace cycfi::elements
{
   constexpr auto min_port_size = 32;

   ////////////////////////////////////////////////////////////////////////////
   // port_base class implementation
   ////////////////////////////////////////////////////////////////////////////
   void port_base::draw(context const& ctx)
   {
      auto state = ctx.canvas.new_state();
      ctx.canvas.add_rect(ctx.bounds);
      ctx.canvas.clip();
      proxy_base::draw(ctx);
   }

   /**
    * @brief
    *    Utility to find the bounds established by the innermost port given a
    *    child context. If there is none, returns ctx.view_bounds().
    *
    *    This utility function searches for the bounds of the innermost port.
    *    If no port is found, the function returns the view's bounds.
    *
    * @param ctx
    *    The context of the child element for which the port bounds are being
    *    determined. This context carries information about the current state
    *    of the UI, including any enclosing ports.
    *
    * @return
    *    The rectangular bounds defined by the innermost port affecting the
    *    given context, or the view bounds if no such port exists.
    */
   rect get_port_bounds(context const& ctx)
   {
      if (auto pctx = find_parent_context<port_base*>(ctx))
         return pctx->bounds;
      return ctx.view_bounds();
   }

   view_limits port_element::limits(basic_context const& ctx) const
   {
      view_limits e_limits = subject().limits(ctx);
      return {{min_port_size, min_port_size}, e_limits.max};
   }

   void port_element::prepare_subject(context& ctx)
   {
      view_limits    e_limits          = subject().limits(ctx);
      double         elem_width        = e_limits.min.x;
      double         elem_height       = e_limits.min.y;
      double         available_width   = ctx.parent->bounds.width();
      double         available_height  = ctx.parent->bounds.height();

      ctx.bounds.left -= (elem_width - available_width) * _halign;
      ctx.bounds.width(elem_width);
      ctx.bounds.top -= (elem_height - available_height) * _valign;
      ctx.bounds.height(elem_height);

      subject().layout(ctx);
   }

   view_limits vport_element::limits(basic_context const& ctx) const
   {
      view_limits e_limits = subject().limits(ctx);
      return {{e_limits.min.x, min_port_size}, e_limits.max};
   }

   void vport_element::prepare_subject(context& ctx)
   {
      view_limits e_limits = subject().limits(ctx);
      double elem_height = e_limits.min.y;
      double available_height = ctx.parent->bounds.height();

      ctx.bounds.top -= (elem_height - available_height) * _valign;
      ctx.bounds.height(elem_height);

      subject().layout(ctx);
   }

   view_limits hport_element::limits(basic_context const& ctx) const
   {
      view_limits e_limits = subject().limits(ctx);
      return {{min_port_size, e_limits.min.y}, e_limits.max};
   }

   void hport_element::prepare_subject(context& ctx)
   {
      view_limits e_limits = subject().limits(ctx);
      double elem_width = e_limits.min.x;
      double available_width = ctx.parent->bounds.width();

      ctx.bounds.left -= (elem_width - available_width) * _halign;
      ctx.bounds.width(elem_width);

      subject().layout(ctx);
   }

   /**
    * @brief
    *    Finds the nearest scrollable context in the hierarchy.
    *
    *    This function traverses the context hierarchy starting from the
    *    given context (`ctx_`), searching for the nearest parent (or the
    *    context itself) that contains a scrollable element. It utilizes the
    *    `find_element` function to locate a `scrollable` instance within the
    *    current context's element.
    *
    *    If a scrollable element is found, a `scrollable_context` struct is
    *    returned, containing pointers to both the context and the scrollable
    *    element. If no scrollable element is found in the hierarchy, the
    *    function returns a `scrollable_context` with null pointers.
    *
    * @param ctx_
    *    The starting context for the search.
    *
    * @return
    *    A `scrollable_context` struct with pointers to the found context and
    *    scrollable element. If not found, both pointers in the struct will
    *    be null.
    */
   scrollable::scrollable_context scrollable::find(context const& ctx_)
   {
      auto const* ctx = &ctx_;
      while (ctx && ctx->element)
      {
         auto* sp = find_element<scrollable*>(ctx->element);
         if (sp)
            return {ctx, sp};
         else
            ctx = ctx->parent;
      }
      return {0, 0};
   }

   ////////////////////////////////////////////////////////////////////////////
   // scroller_base class implementation
   ////////////////////////////////////////////////////////////////////////////
   namespace
   {
      void draw_scrollbar_fill(canvas& _canvas, rect r, color fill_color)
      {
         _canvas.begin_path();
         _canvas.add_rect(r);
         _canvas.fill_style(fill_color);
         _canvas.fill();
      }

      void draw_scrollbar(
         canvas& _canvas, rect b, float radius,
         color outline_color, color fill_color, point mp,
         bool is_tracking
      )
      {
         _canvas.begin_path();
         _canvas.add_round_rect(b, radius);
         _canvas.fill_style(fill_color);

         if (is_tracking || _canvas.hit_test(mp))
            _canvas.fill_style(fill_color.opacity(0.8));

         _canvas.fill_preserve();

         _canvas.line_width(0.5);
         _canvas.stroke_style(outline_color);
         _canvas.stroke();
      }
   }

   /**
    * @brief
    *    Draws the scrollbar for the scroller_base.
    *
    * @param ctx
    *    The drawing context, containing the canvas and other relevant
    *    drawing information.
    *
    * @param info
    *    A `scrollbar_info` struct containing details about the scrollbar's
    *    bounds, position, and extent.
    *
    * @param mp
    *    The current mouse position.
    */
   void scroller_base::draw_scroll_bar(context const& ctx, scrollbar_info const& info, point mp)
   {
      theme const& thm = get_theme();
      auto state = ctx.canvas.new_state();

      float x = info.bounds.left;
      float y = info.bounds.top;
      float w = info.bounds.width();
      float h = info.bounds.height();

      draw_scrollbar_fill(ctx.canvas, info.bounds, thm.scrollbar_color);

      if (w > h)
      {
         w *= w / info.extent;
         clamp_min(w, 20);
         x += info.pos * (info.bounds.width()-w);
      }
      else
      {
         h *= h / info.extent;
         clamp_min(h, 20);
         y += info.pos * (info.bounds.height()-h);
      }

      draw_scrollbar(ctx.canvas, rect{x, y, x+w, y+h}, thm.scrollbar_width / 3,
         thm.frame_color.opacity(0.5), thm.scrollbar_color.opacity(0.4), mp,
         _tracking == ((w > h)? tracking_h : tracking_v));
   }

   /**
    * @brief
    *    Calculates the position and size of the scrollbar.
    *
    *    This function determines the position and size of the scrollbar
    *    within the scroller_base, based on the current scrolling information
    *    provided by `info`.
    *
    * @param ctx
    *    The drawing context.
    *
    * @param info
    *    A `scrollbar_info` struct containing details about the current
    *    scroll state, including the bounds of the scroller_base, the
    *    position of the scrollbar, and the total extent of the content.
    *
    * @return
    *    A `rect` representing the position and size of the scrollbar.
    */
   rect scroller_base::scroll_bar_position(context const& /* ctx */, scrollbar_info const& info)
   {
      float x = info.bounds.left;
      float y = info.bounds.top;
      float w = info.bounds.width();
      float h = info.bounds.height();

      if (w > h)
      {
         w *= w  / info.extent;
         clamp_min(w, 20);
         x += info.pos * (info.bounds.width()-w);
      }
      else
      {
         h *= h / info.extent;
         clamp_min(h, 20);
         y += info.pos * (info.bounds.height()-h);
      }
      return rect{x, y, x+w, y+h};
   }

   view_limits scroller_base::limits(basic_context const& ctx) const
   {
      view_limits e_limits = subject().limits(ctx);
      auto min_x = allow_hscroll() ? min_port_size : e_limits.min.x;
      auto min_y = allow_vscroll() ? min_port_size : e_limits.min.y;
      auto max_x = std::max(min_x, e_limits.max.x);
      auto max_y = std::max(min_y, e_limits.max.y);
      return view_limits{{min_x, min_y}, {max_x, max_y}};
   }

   void scroller_base::prepare_subject(context& ctx)
   {
      view_limits e_limits = subject().limits(ctx);

      if (allow_vscroll())
      {
         double elem_height = e_limits.min.y;
         double available_height = ctx.parent->bounds.height();

         if (elem_height <= available_height)
            valign(0.0);
         else
            ctx.bounds.top -= (elem_height - available_height) * valign();
         ctx.bounds.height(elem_height);
      }

      if (allow_hscroll())
      {
         double elem_width = e_limits.min.x;
         double available_width = ctx.parent->bounds.width();

         if (elem_width <= available_width)
            halign(0.0);
         else
            ctx.bounds.left -= (elem_width - available_width) * halign();
         ctx.bounds.width(elem_width);
      }
      subject().layout(ctx);
   }

   element* scroller_base::hit_test(context const& ctx, point p, bool leaf, bool control)
   {
      return element::hit_test(ctx, p, leaf, control);
   }

   // Hide the scroll bar if the content size, multiplied by this threshold,
   // is larger than the available size.
   constexpr auto scroll_bar_visibility_threshold = 0.99;

   scroller_base::scrollbar_bounds
   scroller_base::get_scrollbar_bounds(context const& ctx)
   {
      scrollbar_bounds r;
      view_limits e_limits = subject().limits(ctx);
      theme const& thm = get_theme();

      r.has_h = allow_hscroll() &&
         e_limits.min.x * scroll_bar_visibility_threshold >= ctx.bounds.width();
      r.has_v = allow_vscroll() &&
         e_limits.min.y * scroll_bar_visibility_threshold >= ctx.bounds.height();

      if (r.has_v)
      {
         r.vscroll_bounds = rect{
            ctx.bounds.left + ctx.bounds.width() - thm.scrollbar_width,
            ctx.bounds.top,
            ctx.bounds.right,
            ctx.bounds.bottom - (r.has_h ? thm.scrollbar_width : 0)
         };
      }
      else
      {
         valign(0.0);
      }

      if (r.has_h)
      {
         r.hscroll_bounds = rect{
            ctx.bounds.left,
            ctx.bounds.top + ctx.bounds.height() - thm.scrollbar_width,
            ctx.bounds.right - (r.has_v ? thm.scrollbar_width : 0),
            ctx.bounds.bottom
         };
      }
      else
      {
         halign(0.0);
      }
      return r;
   }

   void scroller_base::draw(context const& ctx)
   {
      port_element::draw(ctx);

      if (has_scrollbars())
      {
         scrollbar_bounds sb = get_scrollbar_bounds(ctx);
         view_limits e_limits = subject().limits(ctx);
         point mp = ctx.cursor_pos();

         if (sb.has_v)
            draw_scroll_bar(ctx, {valign(), e_limits.min.y, sb.vscroll_bounds}, mp);

         if (sb.has_h)
            draw_scroll_bar(ctx, {halign(), e_limits.min.x, sb.hscroll_bounds}, mp);
      }
   }

   bool scroller_base::scroll(context const& ctx, point dir, point p)
   {
      view_limits e_limits = subject().limits(ctx);
      bool redraw = false;

      if (allow_hscroll())
      {
         double dx = (-dir.x / (e_limits.min.x - ctx.bounds.width()));
         if ((dx > 0 && halign() < 1.0) || (dx < 0 && halign() > 0.0))
         {
            double alx = halign() + dx;
            clamp(alx, 0.0, 1.0);
            halign(alx);
            redraw = true;
         }
      }

      if (allow_vscroll())
      {
         double dy = (-dir.y / (e_limits.min.y - ctx.bounds.height()));
         if ((dy > 0 && valign() < 1.0) || (dy < 0 && valign() > 0.0))
         {
            double aly = valign() + dy;
            clamp(aly, 0.0, 1.0);
            valign(aly);
            redraw = true;
         }
      }

      if (redraw)
      {
         on_scroll(point(halign(), valign()));
         ctx.view.refresh(ctx);
      }
      return port_element::scroll(ctx, dir, p) || redraw;
   }

   /**
    * @brief
    *    Sets the scroll alignment of the scroller_base.
    *
    *    This function adjusts the scroll alignment of the scroller_base
    *    object based on the specified point `p`, considering horizontal and
    *    vertical scrolling capabilities, and adjusts the scroll alignment
    *    accordingly.
    *
    *    If horizontal scrolling is allowed (`allow_hscroll()` returns true),
    *    it sets the horizontal alignment (scroll alignment) to the
    *    x-coordinate of the point `p`. Similarly, if vertical scrolling is
    *    allowed (`allow_vscroll()` returns true), it sets the vertical
    *    alignment (scroll alignment) to the y-coordinate of the point `p`.
    *
    *    Take note that alignment values are clamped to the range [0.0, 1.0].
    *
    * @param p
    *    The point representing the new scroll alignment, where `p.x` is the
    *    new horizontal alignment and `p.y` is the new vertical alignment.
    */
   void scroller_base::set_alignment(point p)
   {
      if (allow_hscroll())
         halign(p.x);
      if (allow_vscroll())
         valign(p.y);
   }

   bool scroller_base::click(context const& ctx, mouse_button btn)
   {
      if (btn.state == mouse_button::left && has_scrollbars())
      {
         if (btn.down)
         {
            _tracking = start;
            if (reposition(ctx, btn.pos))
               return true;
         }
         _tracking = none;
         refresh(ctx);
      }
      return port_element::click(ctx, btn);
   }

   void scroller_base::drag(context const& ctx, mouse_button btn)
   {
      if (btn.state == mouse_button::left &&
         (_tracking == none || !reposition(ctx, btn.pos)))
         port_element::drag(ctx, btn);
   }

   bool scroller_base::reposition(context const& ctx, point p)
   {
      if (!has_scrollbars())
         return false;

      scrollbar_bounds  sb = get_scrollbar_bounds(ctx);
      view_limits       e_limits = subject().limits(ctx);

      auto valign_ = [&](double align)
      {
         clamp(align, 0.0, 1.0);
         valign(align);
         on_scroll(point(halign(), align));
         ctx.view.refresh(ctx);
      };

      auto halign_ = [&](double align)
      {
         clamp(align, 0.0, 1.0);
         halign(align);
         on_scroll(point(align, valign()));
         ctx.view.refresh(ctx);
      };

      if (sb.has_v)
      {
         // vertical scroll
         rect b = scroll_bar_position(
            ctx, {valign(), e_limits.min.y, sb.vscroll_bounds});

         if (_tracking == start)
         {
            if (b.includes(p))
            {
               // start tracking scroll-box
               _offset = point{p.x-b.left, p.y-b.top};
               _tracking = tracking_v;
            }
            else if (sb.vscroll_bounds.includes(p))
            {
               // page up or down
               double page = b.height() / sb.vscroll_bounds.height();
               if (p.y < b.top)
               {
                  valign_(valign() - page);
                  return true;
               }
               else if (p.y > b.bottom)
               {
                  valign_(valign() + page);
                  return true;
               }
            }
         }

         // continue tracking scroll-box
         if (_tracking == tracking_v)
         {
            p.y -= _offset.y + ctx.bounds.top;
            valign_(p.y / (sb.vscroll_bounds.height() - b.height()));
            return true;
         }
      }

      if (sb.has_h)
      {
         // horizontal scroll
         rect b = scroll_bar_position(
            ctx, {halign(), e_limits.min.x, sb.hscroll_bounds});

         if (_tracking == start)
         {
            // start tracking scroll-box
            if (b.includes(p))
            {
               _offset = point{p.x-b.left, p.y-b.top};
               _tracking = tracking_h;
            }
            else if (sb.hscroll_bounds.includes(p))
            {
               // page left or right
               double page = b.width() / sb.vscroll_bounds.width();
               if (p.x < b.left)
               {
                  halign_(valign() - page);
                  return true;
               }
               else if (p.x > b.right)
               {
                  halign_(valign() + page);
                  return true;
               }
            }
         }

         // continue tracking scroll-box
         if (_tracking == tracking_h)
         {
            p.x -= _offset.x + ctx.bounds.left;
            halign_(p.x / (sb.hscroll_bounds.width() - b.width()));
            return true;
         }
      }

      return false;
   }

   bool scroller_base::cursor(context const& ctx, point p, cursor_tracking status)
   {
      if (has_scrollbars())
      {
         scrollbar_bounds sb = get_scrollbar_bounds(ctx);
         if (sb.hscroll_bounds.includes(p) || sb.vscroll_bounds.includes(p))
         {
            ctx.view.refresh(ctx);
            set_cursor(cursor_type::arrow);
            return true;
         }
         ctx.view.refresh(ctx);
      }
      return port_element::cursor(ctx, p, status);
   }

   bool scroller_base::wants_control() const
   {
      return true;
   }

   bool scroller_base::scroll_into_view(context const& ctx, rect r)
   {
      rect bounds = ctx.bounds;
      theme const& thm = get_theme();

      if (has_scrollbars())
      {
         scrollbar_bounds sb = get_scrollbar_bounds(ctx);

         if (sb.has_h)
            bounds.right -= thm.scrollbar_width;
         if (sb.has_v)
            bounds.bottom -= thm.scrollbar_width;
      }

      if (!bounds.includes(r))
      {
         // r is not in view, we need to scroll
         point dp;

         if (allow_vscroll())
         {
            if (r.top < bounds.top)
               dp.y = bounds.top-r.top;
            else if (r.bottom > bounds.bottom)
               dp.y = bounds.bottom-r.bottom;
         }

         if (allow_hscroll())
         {
            if (r.left < bounds.left)
               dp.x = bounds.left-r.left;
            else if (r.right > bounds.right)
               dp.x = bounds.right-r.right;
         }

         return scroll(ctx, dp, ctx.cursor_pos());
      }
      return false;
   }

   bool scroller_base::key(context const& ctx, key_info k)
   {
      auto valign_ = [&](double align)
      {
         clamp(align, 0.0, 1.0);
         valign(align);
         ctx.view.refresh(ctx);
      };

      bool handled = proxy_base::key(ctx, k);
      if (!handled && (k.action == key_action::press || k.action == key_action::repeat))
      {
         switch (k.key)
         {
            case key_code::home:
               valign_(0);
               handled = true;
               break;

            case key_code::end:
               valign_(1);
               handled = true;
               break;

            case key_code::page_up:
            case key_code::page_down:
            {
               view_limits e_limits = subject().limits(ctx);
               scrollbar_bounds sb = get_scrollbar_bounds(ctx);
               rect b = scroll_bar_position(
                  ctx, {valign(), e_limits.min.y, sb.vscroll_bounds});
               double page = b.height() / sb.vscroll_bounds.height();
               valign_(valign() + ((k.key == key_code::page_down) ? page : -page));
               handled = true;
               break;
            }

            default:
               break;
         }
      }
      return handled;
   }
}
