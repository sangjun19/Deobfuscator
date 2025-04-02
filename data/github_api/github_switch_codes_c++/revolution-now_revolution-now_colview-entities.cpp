/****************************************************************
**colview-entities.cpp
*
* Project: Revolution Now
*
* Created by dsicilia on 2020-01-12.
*
* Description: The various UI sections/entities in Colony view.
*
*****************************************************************/
#include "colview-entities.hpp"

// Revolution Now
#include "co-wait.hpp"
#include "colony-buildings.hpp"
#include "colony-mgr.hpp"
#include "colony.hpp"
#include "colview-buildings.hpp"
#include "colview-land.hpp"
#include "colview-population.hpp"
#include "colview-production.hpp"
#include "commodity.hpp"
#include "damaged.hpp"
#include "equip.hpp"
#include "iengine.hpp"
#include "igui.hpp"
#include "interrupts.hpp"
#include "land-production.hpp"
#include "missionary.hpp"
#include "on-map.hpp"
#include "plow.hpp"
#include "production.hpp"
#include "render.hpp"
#include "road.hpp"
#include "screen.hpp" // FIXME: remove
#include "text.hpp"
#include "tiles.hpp"
#include "ts.hpp"
#include "unit-flag.hpp"
#include "unit-mgr.hpp"
#include "unit-ownership.hpp"
#include "unit-transformation.hpp"
#include "views.hpp"

// config
#include "config/colony.rds.hpp"
#include "config/unit-type.hpp"
#include "config/unit-type.rds.hpp"

// ss
#include "ss/colonies.hpp"
#include "ss/players.rds.hpp"
#include "ss/ref.hpp"
#include "ss/terrain.hpp"
#include "ss/units.hpp"

// render
#include "render/extra.hpp"
#include "render/renderer.hpp"

// rds
#include "rds/switch-macro.hpp"

// refl
#include "refl/query-enum.hpp"
#include "refl/to-str.hpp"

// base
#include "base/conv.hpp"
#include "base/logger.hpp"
#include "base/maybe-util.hpp"

using namespace std;

namespace rn {

// Use this as the vtable key function.
void ColonySubView::update_this_and_children() {}

namespace {

using ::gfx::pixel;
using ::gfx::point;
using ::gfx::rect;
using ::gfx::size;

/****************************************************************
** Constants
*****************************************************************/
constexpr W kCommodityTileWidth = kCommodityTileSize.w;

// TODO: temporary.
auto const BROWN_COLOR =
    pixel::parse_from_hex( "f1cf81" ).value().shaded( 14 );

/****************************************************************
** Globals
*****************************************************************/
struct ColViewComposited {
  ColonyId id;
  Delta canvas_size;
  unique_ptr<ColonySubView> top_level;
  unordered_map<e_colview_entity, ColonySubView*> entities;
};

// FIXME
ColViewComposited g_composition;
ColonyProduction g_production;

string_view constexpr kReduceStockadeThreeMsg =
    "We cannot willingly reduce the population of a colony with "
    "a stockade below three.";

/****************************************************************
** Helpers
*****************************************************************/
Cargo to_cargo( ColViewObject const& o ) {
  switch( o.to_enum() ) {
    using e = ColViewObject::e;
    case e::unit:
      return Cargo::unit{ o.get<ColViewObject::unit>().id };
    case e::commodity:
      return Cargo::commodity{
        o.get<ColViewObject::commodity>().comm };
  }
}

ColViewObject from_cargo( Cargo const& o ) {
  return overload_visit<ColViewObject>(
      o, //
      []( Cargo::unit u ) {
        return ColViewObject::unit{ .id = u.id };
      },
      []( Cargo::commodity const& c ) {
        return ColViewObject::commodity{ .comm = c.obj };
      } );
}

// Returns whether the action should be rejected, which will
// happen if the user tries to reduce the population of a colony
// with a stockade below three.
bool check_stockade_3( Colony const& colony ) {
  bool const has_stockade_or_higher = colony_has_building_level(
      colony, e_colony_building::stockade );
  bool const should_reject =
      has_stockade_or_higher && colony_population( colony ) <= 3;
  return should_reject;
}

// Returns whether the action should be rejected.
wait<bool> check_abandon( Colony const& colony, IGui& gui ) {
  if( colony_population( colony ) > 1 ) co_return false;
  YesNoConfig const config{
    .msg = "Shall we abandon this colony, Your Excellency?",
    .yes_label      = "Yes, it is God's will.",
    .no_label       = "Never!  That would be folly.",
    .no_comes_first = true,
  };
  maybe<ui::e_confirm> res =
      co_await gui.optional_yes_no( config );
  co_return ( res != ui::e_confirm::yes );
}

maybe<string> check_seige() {
  // TODO: check if the colony is under seige; in that case
  // colonists are not allowed to move from the fields to the
  // gates.
  return nothing;
}

// This function is called when we are dragging a unit who is
// working inside the colony to a place outside the colony,
// meaning that it will be removed from the colony. Before we do
// that, we need to check if we are abandoning this colony by
// doing so. In that case, we should allow the user to salvage
// any horses, tools, or muskets in the store to transform the
// unit into a scout, pioneer, or soldier. Otherwise, the colony
// would be abandoned and those commodities would be lost. In
// other words, if we didn't do this, there would be no way to
// e.g. found a colony with a scout (leaving 50 horses in the
// colony) and then abandon it reproducing the same scout (this
// is useful e.g. when a scout finds treasure in a remote part of
// the map and needs to move it into a colony for transport by
// the king).
wait<maybe<ColonyEquipOption>> ask_transorm_unit_on_leave(
    TS& ts, Colony const& colony, Unit const& unit ) {
  if( colony_population( colony ) > 1 ) co_return nothing;
  ChoiceConfig config{
    .msg = fmt::format(
        "As this action would abandon the colony and discard "
        "all of its contents, shall we equip this [{}] "
        "before proceeding?",
        unit.desc().name ),
    .options = {},
    .sort    = false,
  };
  vector<ColonyEquipOption> const equip_opts =
      colony_equip_options( colony, unit.composition() );
  // We should always at least have the option to leave the unit
  // unchanged.
  CHECK_GT( equip_opts.size(), 0U );
  if( equip_opts.size() == 1 ) {
    // We have only the option to keep the unit the same. Since
    // the unit was original a colonist with no modifiers, we can
    // say that this one option should have an identical composi-
    // tion (e.g., we're not keeping the unit type the same but
    // increasing its tool count).
    CHECK( equip_opts[0].new_comp == unit.composition() );
    co_return nothing;
  }
  static string const kNoChangesKey = "no changes";
  config.options.push_back(
      { .key = kNoChangesKey, .display_name = "No Changes." } );
  for( int idx = 0; idx < int( equip_opts.size() ); ++idx ) {
    ColonyEquipOption const& equip_opt = equip_opts[idx];
    if( equip_opt.new_comp == unit.composition() ) continue;
    ChoiceConfigOption option{
      .key          = fmt::to_string( idx ),
      .display_name = colony_equip_description( equip_opt ) };
    config.options.push_back( std::move( option ) );
  }
  maybe<string> const choice =
      co_await ts.gui.optional_choice( config );
  if( !choice.has_value() ) co_return nothing;
  if( choice == kNoChangesKey ) co_return nothing;
  // Assume we have an index into the ColonyEquipOptions vector.
  UNWRAP_CHECK( chosen_idx, base::from_chars<int>( *choice ) );
  CHECK_GE( chosen_idx, 0 );
  CHECK_LT( chosen_idx, int( equip_opts.size() ) );
  // This will change the unit type.
  co_return equip_opts[chosen_idx];
}

// This is called whenever a unit is changed in any way (either
// its type/composition or cargo is changed). In the OG, various
// unit types react differently to that.
void adjust_mv_points_from_drag( Unit& unit ) {
  if( unit.desc().ship )
    // For a ship we will not clear movement points under any
    // circumstances when loading unloading. This is because the
    // game will already forfeight the ship's movement points
    // when it moves into a colony port, so there is no need for
    // an additional penalty.
    return;

  if( unit.type() == e_unit_type::wagon_train ) {
    // In the OG, a wagon train ends its turn when it moves into
    // a colony square, just like a ship. If we were replicating
    // that here then we would do as we did for the ship above
    // and just do nothing. However, in this game we don't have
    // the wagon train forfeight its movement points upon en-
    // tering a colony because 1) it has proven to be an annoying
    // feature, 2) no other land unit has this behavior, and 3)
    // there doesn't seem to be any logical reason for it.
    //
    // That said, it is possible that the original game imposed
    // this behavior (for both wagon trains and ships) in order
    // to prevent such a unit from moving into a colony and load-
    // ing/unloading, then continuing to move, all in one turn.
    // So, in order to uphold the spirit of that rule, we will
    // forfeight the wagon train's movement points not when it
    // pass through a colony per se, but when it loads or unloads
    // any goods within it. Actually, not quite... if we did that
    // then it might lead to some player confusion since if, on a
    // given turn, the player activates a wagon train (that has
    // not yet moved this turn) then loads something into it
    // planning to move it, then they will not be able to move it
    // that turn. So what we will do is to only forfeight the
    // movement points when the wagon train has already partially
    // moved this turn. That will allow the player to load a
    // wagon train at the top of its turn and still move it, but
    // won't allow the player to move the wagon train into a
    // colony, load/unload, and keep moving in the same turn
    // (something the OG probably wanted to prevent).
    //
    // This seems to make more sense and gives a good balance be-
    // tween a sensible player experience and upholding the goals
    // of the OG.
    //
    // Note that we don't replicate this new behavior for the
    // ship; as stated above, we just keep the OG's behavior
    // there. In the case of a ship it makes more intuitive sense
    // that the ship would have to lose all of its movement
    // points when moving into a port since it has to e.g. slow
    // down and maneuver. And since we're consuming its movement
    // points when it goes into port, we don't impose a further
    // penalty of consuming its movement points when loading or
    // unloading, since again we have already upheld the likely
    // desire of the OG in preventing a ship from moving into a
    // colony, loading/unloading, and continuing to move all in
    // the same turn.
    if( !unit.has_full_mv_points() ) unit.forfeight_mv_points();
    return;
  }

  // In the OG, when a non-cargo-holding unit changes type by ei-
  // ther gaining or losing some commodities or modifiers then it
  // will lose its movement points that turn.
  unit.forfeight_mv_points();
}

/****************************************************************
** Entities
*****************************************************************/
class TitleBar : public ui::View, public ColonySubView {
 public:
  static unique_ptr<TitleBar> create( IEngine& engine, SS& ss,
                                      TS& ts, Player& player,
                                      Colony& colony,
                                      Delta size ) {
    return make_unique<TitleBar>( engine, ss, ts, player, colony,
                                  size );
  }

  TitleBar( IEngine& engine, SS& ss, TS& ts, Player& player,
            Colony& colony, Delta size )
    : ColonySubView( engine, ss, ts, player, colony ),
      size_( size ) {}

  Delta delta() const override { return size_; }

  // Implement IDraggableObjectsView.
  maybe<int> entity() const override {
    return static_cast<int>( e_colview_entity::title_bar );
  }

  ui::View& view() noexcept override { return *this; }
  ui::View const& view() const noexcept override {
    return *this;
  }

  string title() const {
    auto const& colony = ss_.colonies.colony_for( colony_.id );
    return fmt::format( "{}, population {}", colony.name,
                        colony_population( colony ) );
  }

  void draw( rr::Renderer& renderer,
             Coord coord ) const override {
    rr::Painter painter = renderer.painter();
    painter.draw_solid_rect( bounds( coord ),
                             gfx::pixel::wood() );
    rr::Typer typer = renderer.typer( rr::TextLayout{} );
    typer.set_color( pixel::banana() );
    typer.set_position(
        gfx::centered_in( typer.dimensions_for_line( title() ),
                          bounds( coord ).to_gfx() ) );
    typer.write( title() );
  }

 private:
  Delta size_;
};

class MarketCommodities
  : public ui::View,
    public ColonySubView,
    public IDragSource<ColViewObject>,
    public IDragSourceUserEdit<ColViewObject>,
    public IDragSink<ColViewObject> {
 public:
  static unique_ptr<MarketCommodities> create( IEngine& engine,
                                               SS& ss, TS& ts,
                                               Player& player,
                                               Colony& colony,
                                               W block_width ) {
    return make_unique<MarketCommodities>(
        engine, ss, ts, player, colony, block_width );
  }

  MarketCommodities( IEngine& engine, SS& ss, TS& ts,
                     Player& player, Colony& colony,
                     W block_width )
    : ColonySubView( engine, ss, ts, player, colony ),
      block_width_( block_width ) {}

  Delta delta() const override {
    return Delta{
      block_width_ * SX{ refl::enum_count<e_commodity> },
      1 * 32 };
  }

  // Implement IDraggableObjectsView.
  maybe<int> entity() const override {
    return static_cast<int>( e_colview_entity::commodities );
  }

  ui::View& view() noexcept override { return *this; }
  ui::View const& view() const noexcept override {
    return *this;
  }

  // Offset within a block that the commodity icon should be dis-
  // played.
  Delta rendered_commodity_offset() const {
    Delta res;
    res.h = 3;
    res.w = ( block_width_ - kCommodityTileWidth ) / 2;
    if( res.w < 0 ) res.w = 0;
    return res;
  }

  void draw( rr::Renderer& renderer,
             Coord coord ) const override {
    rr::Painter painter = renderer.painter();
    auto comm_it        = refl::enum_values<e_commodity>.begin();
    auto label          = CommodityLabel::quantity{ 0 };
    Coord pos           = coord;
    auto const& colony  = ss_.colonies.colony_for( colony_.id );
    int const warehouse_limit =
        colony_warehouse_capacity( colony );
    bool const has_custom_house =
        colony.buildings[e_colony_building::custom_house];
    for( int i = 0; i < kNumCommodityTypes; ++i ) {
      auto rect =
          Rect::from( pos, Delta{ .w = block_width_, .h = 32 } );
      // FIXME: this color should be deduped with the one in the
      // harbor view.
      static gfx::pixel const bg_color = gfx::pixel{
        .r = 0x90, .g = 0x90, .b = 0xc0, .a = 0xff };
      painter.draw_solid_rect( rect, bg_color );
      painter.draw_empty_rect(
          rect, rr::Painter::e_border_mode::in_out,
          pixel::black() );
      label.value = colony.commodities[*comm_it];
      // Regarding the choice of colors: the OG will color the
      // label red when it exceeds the warehouse capacity of the
      // colony regardless of whether the commodity is being sold
      // by the custom house. But in this game we will suppress
      // the red colors if the commodity is being sold by the
      // custom house (even if it is over capacity) since that
      // arguably gives a better indication to the player of what
      // will and won't spoil (note that when a custom house is
      // selling a commodity it cannot spoil, since the selling
      // happens before the spoiling each turn).
      //
      // The reason that we're checking to see if the custom
      // house is present here (instead of just checking the
      // custom house state per commodity) is because it is pos-
      // sible that a colony's custom house could go away after
      // construction, e.g. if it is destroyed by the indians or
      // if the player removes it via cheat mode.
      if( has_custom_house && colony_.custom_house[*comm_it] )
        label.colors = e_commodity_label_render_colors::
            custom_house_selling;
      else if( config_colony.warehouses
                   .commodities_with_warehouse_limit[*comm_it] &&
               colony_.commodities[*comm_it] > warehouse_limit )
        label.colors =
            e_commodity_label_render_colors::over_limit;
      else
        label.colors = e_commodity_label_render_colors::standard;
      // When we drag a commodity we want the effect to be that
      // the commodity icon is still drawn (because it is a kind
      // of label for buckets), but we want the quantity to
      // render as zero to reflect the fact that the player has
      // removed those from the colony store.
      if( *comm_it == dragging_.member( &Commodity::type ) )
        label.value = 0;
      render_commodity_annotated_16(
          renderer,
          rect.upper_left() + rendered_commodity_offset(),
          *comm_it,
          CommodityRenderStyle{ .label  = label,
                                .dulled = false } );
      pos.x += block_width_;
      comm_it++;
    }
  }

  int quantity_of( e_commodity type ) const {
    return colony_.commodities[type];
  }

  maybe<DraggableObjectWithBounds<ColViewObject>> object_here(
      Coord const& coord ) const override {
    if( !coord.is_inside( bounds( {} ) ) ) return nothing;
    auto sprite_scale =
        Delta{ .w = SX{ block_width_ }, .h = SY{ 32 } };
    auto box_upper_left =
        ( coord / sprite_scale ) * sprite_scale;
    auto idx = ( coord / sprite_scale - Coord{} ).w;
    UNWRAP_CHECK( type, commodity_from_index( idx ) );
    int quantity = quantity_of( type );
    // NOTE: we don't enforce that the quantity be greater than
    // zero here, instead we do that in try_drag. That way we can
    // still recognize what is under the cursor even if there is
    // zero quantity of it.
    return DraggableObjectWithBounds<ColViewObject>{
      .obj    = ColViewObject::commodity{ Commodity{
           .type = type, .quantity = quantity } },
      .bounds = Rect::from(
          box_upper_left + rendered_commodity_offset(),
          Delta{ .w = 1, .h = 1 } * kCommodityTileSize ) };
  }

  bool try_drag( ColViewObject const& o,
                 Coord const& where ) override {
    UNWRAP_CHECK( [c], o.get_if<ColViewObject::commodity>() );
    if( c.quantity == 0 ) return false;
    // Sanity checks.
    UNWRAP_CHECK( here, object_here( where ) );
    UNWRAP_CHECK( comm_at_source,
                  here.obj.get_if<ColViewObject::commodity>() );
    Commodity dragged_c = comm_at_source.comm;
    CHECK( dragged_c.type == c.type );
    // Could be less if the destination has limited space and
    // has edited `o` to be less in quantity than the source.
    CHECK( c.quantity <= dragged_c.quantity );
    // End sanity checks.
    dragging_ = c;
    return true;
  }

  void cancel_drag() override { dragging_ = nothing; }

  wait<> disown_dragged_object() override {
    CHECK( dragging_ );
    e_commodity type = dragging_->type;
    int new_quantity = quantity_of( type ) - dragging_->quantity;
    CHECK( new_quantity >= 0 );
    colony_.commodities[type] = new_quantity;
    co_return;
  }

  maybe<CanReceiveDraggable<ColViewObject>> can_receive(
      ColViewObject const& o, int /*from_entity*/,
      Coord const& where ) const override {
    CHECK( where.is_inside( bounds( {} ) ) );
    if( o.holds<ColViewObject::commodity>() )
      return CanReceiveDraggable<ColViewObject>::yes{
        .draggable = o };
    return nothing;
  }

  wait<> drop( ColViewObject const& o,
               Coord const& /*where*/ ) override {
    UNWRAP_CHECK( [c], o.get_if<ColViewObject::commodity>() );
    int q = colony_.commodities[c.type];
    q += c.quantity;
    colony_.commodities[c.type] = q;
    co_return;
  }

  wait<maybe<ColViewObject>> user_edit_object() const override {
    CHECK( dragging_ );
    int min     = 1;
    int max     = dragging_->quantity;
    string text = fmt::format(
        "What quantity of [{}] would you like to move? "
        "({}-{}):",
        lowercase_commodity_display_name( dragging_->type ), min,
        max );
    maybe<int> quantity = co_await ts_.gui.optional_int_input(
        { .msg           = text,
          .initial_value = max,
          .min           = min,
          .max           = max } );
    if( !quantity ) co_return nothing;
    Commodity new_comm = *dragging_;
    new_comm.quantity  = *quantity;
    CHECK( new_comm.quantity > 0 );
    co_return from_cargo( Cargo::commodity{ new_comm } );
  }

 private:
  W block_width_;
  maybe<Commodity> dragging_;
};

class CargoView : public ui::View,
                  public ColonySubView,
                  public IDragSource<ColViewObject>,
                  public IDragSourceUserEdit<ColViewObject>,
                  public IDragSink<ColViewObject>,
                  public IDragSinkUserEdit<ColViewObject>,
                  public IDragSinkCheck<ColViewObject> {
 public:
  static unique_ptr<CargoView> create( IEngine& engine, SS& ss,
                                       TS& ts, Player& player,
                                       Colony& colony,
                                       Delta size ) {
    return make_unique<CargoView>( engine, ss, ts, player,
                                   colony, size );
  }

  CargoView( IEngine& engine, SS& ss, TS& ts, Player& player,
             Colony& colony, Delta size )
    : ColonySubView( engine, ss, ts, player, colony ),
      size_( size ) {}

  Delta delta() const override { return size_; }

  // Implement IDraggableObjectsView.
  maybe<int> entity() const override {
    return static_cast<int>( e_colview_entity::cargo );
  }

  ui::View& view() noexcept override { return *this; }
  ui::View const& view() const noexcept override {
    return *this;
  }

  int max_slots_drawable() const {
    return delta().w / g_tile_delta.w;
  }

  // As usual, coordinate must be relative to upper left corner
  // of this view.
  maybe<pair<bool, int>> slot_idx_from_coord(
      Coord const& c ) const {
    if( !c.is_inside( bounds( {} ) ) ) return nothing;
    if( c.y > 0 + g_tile_delta.h ) return nothing;
    int slot_idx = ( c / g_tile_delta ).distance_from_origin().w;
    bool is_open =
        holder_.has_value() &&
        slot_idx <
            ss_.units.unit_for( *holder_ ).desc().cargo_slots;
    return pair{ is_open, slot_idx };
  }

  // Returned rect is relative to upper left of this view.
  maybe<pair<bool, Rect>> slot_rect_from_idx( int slot ) const {
    if( slot < 0 ) return nothing;
    if( slot >= max_slots_drawable() ) return nothing;
    Coord slot_upper_left =
        Coord{} + Delta{ .w = g_tile_delta.w * slot };
    bool is_open =
        holder_.has_value() &&
        slot < ss_.units.unit_for( *holder_ ).desc().cargo_slots;
    return pair{ is_open,
                 Rect::from( slot_upper_left, g_tile_delta ) };
  }

  void draw( rr::Renderer& renderer,
             Coord coord ) const override {
    rr::Painter painter = renderer.painter();
    painter.draw_empty_rect( bounds( coord ),
                             rr::Painter::e_border_mode::in_out,
                             pixel::black() );
    auto unit = holder_.fmap(
        [&]( UnitId id ) { return ss_.units.unit_for( id ); } );
    for( int idx{ 0 }; idx < max_slots_drawable(); ++idx ) {
      UNWRAP_CHECK( info, slot_rect_from_idx( idx ) );
      auto [is_open, relative_rect] = info;
      Rect rect = relative_rect.as_if_origin_were( coord );
      if( !is_open ) {
        painter.draw_solid_rect(
            rect.shifted_by( Delta{ .w = 1, .h = 0 } ),
            gfx::pixel::wood() );
        continue;
      }

      // FIXME: need to deduplicate this logic with that in
      // the Old World view.
      painter.draw_solid_rect(
          rect, gfx::pixel::wood().highlighted( 4 ) );
      painter.draw_empty_rect(
          rect, rr::Painter::e_border_mode::in_out,
          gfx::pixel::wood() );
      if( dragging_.has_value() && dragging_->slot == idx )
        // If we're draggin the thing in this slot then don't
        // draw it in there.
        continue;
      CargoHold const& hold = unit->cargo();
      switch( auto& v = hold[idx]; v.to_enum() ) {
        case CargoSlot::e::empty:
          break;
        case CargoSlot::e::overflow:
          break;
        case CargoSlot::e::cargo: {
          auto& cargo = v.get<CargoSlot::cargo>();
          overload_visit(
              cargo.contents,
              [&]( Cargo::unit u ) {
                render_unit( renderer, rect.upper_left(),
                             ss_.units.unit_for( u.id ),
                             UnitRenderOptions{} );
              },
              [&]( Cargo::commodity const& c ) {
                render_commodity_annotated_16(
                    renderer,
                    rect.upper_left() +
                        kCommodityInCargoHoldRenderingOffset,
                    c.obj );
              } );
          break;
        }
      }
    }
  }

  void set_unit( maybe<UnitId> unit ) { holder_ = unit; }

  maybe<CanReceiveDraggable<ColViewObject>> can_receive(
      ColViewObject const& o, int from_entity,
      Coord const& where ) const override {
    CHECK( where.is_inside( bounds( {} ) ) );
    if( !holder_ ) return nothing;
    maybe<pair<bool, int>> slot_info =
        slot_idx_from_coord( where );
    if( !slot_info.has_value() ) return nothing;
    auto [is_open, slot_idx] = *slot_info;
    if( !is_open ) return nothing;
    CONVERT_ENTITY( from_enum, from_entity );
    if( from_enum == e_colview_entity::cargo ) {
      // At this point the player is dragging something from one
      // slot to another in the same cargo, which is guaranteed
      // to always be allowed, since when the drag operation will
      // first remove the cargo from the source slot, then when
      // it is dropped, the drop will succeed so long as there is
      // enough space anywhere in the cargo for that cargo, which
      // there always will be, because the cargo originated from
      // within this same cargo.
      return CanReceiveDraggable<ColViewObject>::yes{
        .draggable = o };
    }
    // We are dragging from another source, so we must check to
    // see if we have room for what is being dragged.
    auto& holder = ss_.units.unit_for( *holder_ );
    SWITCH( o ) {
      CASE( unit ) {
        // This is basically to prevent wagon trains from re-
        // ceiving units as cargo.
        if( !holder.desc().can_hold_unit_cargo )
          // It may not be obvious to the player why this is
          // being rejected, so give this one a message.
          return CanReceiveDraggable<ColViewObject>::no_with_msg{
            .msg =
                fmt::format( "[{}] cannot hold units as cargo.",
                             holder.desc().name_plural ) };
        // Note that we allow wagon trains to recieve units at
        // this stage as long as they theoretically fit. In the
        // next stage we will reject that and present a message
        // to the user.
        if( !holder.cargo().fits_somewhere(
                ss_.units, Cargo::unit{ unit.id } ) )
          return nothing;
        return CanReceiveDraggable<ColViewObject>::yes{
          .draggable = o };
      }
      CASE( commodity ) {
        Commodity comm = commodity.comm;
        int const max_quantity =
            holder.cargo().max_commodity_quantity_that_fits(
                comm.type );
        comm.quantity = clamp( comm.quantity, 0, max_quantity );
        if( comm.quantity == 0 ) return nothing;
        return CanReceiveDraggable<ColViewObject>::yes{
          .draggable =
              ColViewObject::commodity{ .comm = comm } };
      }
    }
  }

  // Implement IDragSinkUserEdit.
  wait<maybe<ColViewObject>> user_edit_object(
      ColViewObject const& o, int from_entity,
      Coord const ) const override {
    CONVERT_ENTITY( from_enum, from_entity );
    switch( from_enum ) {
      case e_colview_entity::units_at_gate:
      case e_colview_entity::commodities:
      case e_colview_entity::cargo: //
        co_return o;
      case e_colview_entity::land:
      case e_colview_entity::buildings: {
        // We're dragging an in-colony unit to the gate, so check
        // if it is the last remaining colonist in the colony
        // and, if so, if the user wants to equip it with any
        // horses, tools, or muskets as it leaves and the colony
        // disappears.
        UNWRAP_CHECK( draggable_unit,
                      o.get_if<ColViewObject::unit>() );
        ColViewObject::unit new_draggable_unit = draggable_unit;
        Unit const& unit =
            ss_.units.unit_for( draggable_unit.id );
        maybe<ColonyEquipOption> const equip_options =
            co_await ask_transorm_unit_on_leave( ts_, colony_,
                                                 unit );
        if( equip_options.has_value() )
          new_draggable_unit.transformed = *equip_options;
        co_return new_draggable_unit;
      }
      case e_colview_entity::population:
      case e_colview_entity::title_bar:
      case e_colview_entity::production:
        FATAL( "unexpected source entity." );
    }
  }

  wait<base::valid_or<DragRejection>> sink_check(
      ColViewObject const& o, int from_entity,
      Coord const ) override {
    CHECK( holder_.has_value() );
    // Sanity check.
    if( o.holds<ColViewObject::unit>() ) {
      CHECK( ss_.units.unit_for( *holder_ )
                 .desc()
                 .can_hold_unit_cargo );
    }
    CONVERT_ENTITY( from_enum, from_entity );
    switch( from_enum ) {
      case e_colview_entity::units_at_gate:
      case e_colview_entity::cargo:
      case e_colview_entity::commodities: //
        co_return base::valid;
      case e_colview_entity::land:
      case e_colview_entity::buildings: //
        if( check_stockade_3( colony_ ) )
          co_return DragRejection{
            .reason = string( kReduceStockadeThreeMsg ) };
        if( co_await check_abandon( colony_, ts_.gui ) )
          // If we're rejecting then that means that the player
          // has opted not to abandon the colony, so there is no
          // need to display a reason message.
          co_return DragRejection{ .reason = nothing };
        if( auto msg = check_seige(); msg.has_value() )
          co_return DragRejection{ .reason = *msg };
        co_return base::valid;
      case e_colview_entity::population:
      case e_colview_entity::title_bar:
      case e_colview_entity::production:
        FATAL( "unexpected source entity." );
    }
  }

  wait<> drop( ColViewObject const& o,
               Coord const& where ) override {
    CHECK( holder_ );
    Unit& holder_unit = ss_.units.unit_for( *holder_ );
    // We've added something to the cargo; the OG will clear the
    // orders of the cargo holder for a good player experience.
    holder_unit.clear_orders();
    adjust_mv_points_from_drag( holder_unit );
    auto& cargo_hold = holder_unit.cargo();
    Cargo cargo      = to_cargo( o );
    CHECK( cargo_hold.fits_somewhere( ss_.units, cargo ) );
    UNWRAP_CHECK( slot_info, slot_idx_from_coord( where ) );
    auto [is_open, slot_idx] = slot_info;
    overload_visit(
        cargo, //
        [&, this, slot_idx = slot_idx]( Cargo::unit u ) {
          UNWRAP_CHECK( draggable_unit,
                        o.get_if<ColViewObject::unit>() );
          CHECK( draggable_unit.id == u.id );
          if( draggable_unit.transformed.has_value() ) {
            Unit& to_transform = ss_.units.unit_for( u.id );
            // This will change the unit type and modify colony
            // commodity quantities.
            perform_colony_equip_option(
                ss_, ts_, colony_, to_transform,
                *draggable_unit.transformed );
          }
          UnitOwnershipChanger( ss_, u.id )
              .change_to_cargo( *holder_,
                                /*starting_slot=*/slot_idx );
          // Check if we've abandoned the colony, which could
          // happen if we dragged the last unit working in the
          // colony into the cargo hold.
          if( colony_population( colony_ ) == 0 )
            throw colony_abandon_interrupt{};
        },
        [this,
         slot_idx = slot_idx]( Cargo::commodity const& c ) {
          add_commodity_to_cargo(
              ss_.units, c.obj,
              ss_.units.unit_for( *holder_ ).cargo(), slot_idx,
              /*try_other_slots=*/true );
        } );
    co_return;
  }

  // Returns the rect that bounds the sprite corresponding to the
  // cargo item covered by the given slot.
  maybe<pair<Cargo, Rect>> cargo_item_with_rect(
      int slot ) const {
    maybe<pair<bool, Rect>> slot_rect =
        slot_rect_from_idx( slot );
    if( !slot_rect.has_value() ) return nothing;
    auto [is_open, rect] = *slot_rect;
    if( !is_open ) return nothing;
    maybe<pair<Cargo const&, int>> maybe_cargo =
        ss_.units.unit_for( *holder_ )
            .cargo()
            .cargo_covering_slot( slot );
    if( !maybe_cargo ) return nothing;
    auto const& [cargo, same_slot] = *maybe_cargo;
    CHECK( slot == same_slot );
    return pair{
      cargo,
      overload_visit<Rect>(
          cargo, //
          [rect = rect]( Cargo::unit ) { return rect; },
          [rect = rect]( Cargo::commodity const& ) {
            return Rect::from(
                rect.upper_left() +
                    kCommodityInCargoHoldRenderingOffset,
                kCommodityTileSize );
          } ) };
  }

  maybe<DraggableObjectWithBounds<ColViewObject>> object_here(
      Coord const& where ) const override {
    if( !holder_ ) return nothing;
    maybe<pair<bool, int>> slot_info =
        slot_idx_from_coord( where );
    if( !slot_info ) return nothing;
    auto [is_open, slot_idx] = *slot_info;
    if( !is_open ) return nothing;
    maybe<pair<Cargo, Rect>> cargo_with_rect =
        cargo_item_with_rect( slot_idx );
    if( !cargo_with_rect ) return nothing;
    return DraggableObjectWithBounds<ColViewObject>{
      .obj    = from_cargo( cargo_with_rect->first ),
      .bounds = cargo_with_rect->second };
  }

  // For this one it happens that we need the coordinate instead
  // of the object, since if the object is a commodity we may not
  // be able to find a unique cargo slot that holds that com-
  // modity if there are more than one.
  bool try_drag( ColViewObject const& o,
                 Coord const& where ) override {
    if( !holder_ ) return false;
    maybe<pair<bool, int>> slot_info =
        slot_idx_from_coord( where );
    if( !slot_info ) return false;
    auto [is_open, slot_idx] = *slot_info;
    if( !is_open ) return false;
    dragging_ = Draggable{ .slot = slot_idx, .object = o };
    return true;
  }

  void cancel_drag() override { dragging_ = nothing; }

  wait<> disown_dragged_object() override {
    CHECK( holder_ );
    CHECK( dragging_ );
    Unit& holder_unit = ss_.units.unit_for( *holder_ );
    adjust_mv_points_from_drag( holder_unit );
    // We need to take the stored object instead of just re-
    // trieving it from the slot, because the stored object might
    // have been edited, e.g. the commodity quantity might have
    // been lowered.
    Cargo cargo_to_remove = to_cargo( dragging_->object );
    overload_visit(
        cargo_to_remove,
        [this]( Cargo::unit held ) {
          UnitOwnershipChanger( ss_, held.id ).change_to_free();
        },
        [this]( Cargo::commodity const& to_remove ) {
          UNWRAP_CHECK(
              existing_cargo,
              ss_.units.unit_for( *holder_ )
                  .cargo()
                  .cargo_starting_at_slot( dragging_->slot ) );
          UNWRAP_CHECK(
              existing_comm,
              existing_cargo.get_if<Cargo::commodity>() );
          Commodity reduced_comm = existing_comm.obj;
          CHECK( reduced_comm.type == existing_comm.obj.type );
          CHECK( reduced_comm.type == to_remove.obj.type );
          reduced_comm.quantity -= to_remove.obj.quantity;
          CHECK( reduced_comm.quantity >= 0 );
          rm_commodity_from_cargo(
              ss_.units, ss_.units.unit_for( *holder_ ).cargo(),
              dragging_->slot );
          if( reduced_comm.quantity > 0 )
            add_commodity_to_cargo(
                ss_.units, reduced_comm,
                ss_.units.unit_for( *holder_ ).cargo(),
                dragging_->slot,
                /*try_other_slots=*/false );
        } );
    co_return;
  }

  wait<maybe<ColViewObject>> user_edit_object() const override {
    CHECK( dragging_ );
    UNWRAP_CHECK( cargo_and_rect,
                  cargo_item_with_rect( dragging_->slot ) );
    Cargo const& cargo = cargo_and_rect.first;
    if( !cargo.holds<Cargo::commodity>() )
      co_return from_cargo( cargo );
    // We have a commodity.
    Cargo::commodity const& comm = cargo.get<Cargo::commodity>();
    int min                      = 1;
    int max                      = comm.obj.quantity;
    string text                  = fmt::format(
        "What quantity of [{}] would you like to move? "
                         "({}-{}):",
        lowercase_commodity_display_name( comm.obj.type ), min,
        max );
    maybe<int> quantity = co_await ts_.gui.optional_int_input(
        { .msg           = text,
          .initial_value = max,
          .min           = min,
          .max           = max } );
    if( !quantity ) co_return nothing;
    Commodity new_comm = comm.obj;
    new_comm.quantity  = *quantity;
    CHECK( new_comm.quantity > 0 );
    co_return ColViewObject::commodity{ new_comm };
  }

 private:
  struct Draggable {
    int slot;
    ColViewObject object;
  };

  // FIXME: this gets reset whenever we recomposite. We need to
  // either put this in a global place, or not recreate all of
  // these view objects each time we recomposite (i.e., reuse
  // them).
  maybe<UnitId> holder_;
  Delta size_;
  maybe<Draggable> dragging_;
};

class UnitsAtGateColonyView
  : public ui::View,
    public ColonySubView,
    public IDragSource<ColViewObject>,
    public IDragSink<ColViewObject>,
    public IDragSinkUserEdit<ColViewObject>,
    public IDragSinkCheck<ColViewObject> {
 public:
  static unique_ptr<UnitsAtGateColonyView> create(
      IEngine& engine, SS& ss, TS& ts, Player& player,
      Colony& colony, CargoView* cargo_view, Delta size ) {
    return make_unique<UnitsAtGateColonyView>(
        engine, ss, ts, player, colony, cargo_view, size );
  }

  UnitsAtGateColonyView( IEngine& engine, SS& ss, TS& ts,
                         Player& player, Colony& colony,
                         CargoView* cargo_view, Delta size )
    : ColonySubView( engine, ss, ts, player, colony ),
      cargo_view_( cargo_view ),
      size_( size ) {
    update_this_and_children();
  }

  Delta delta() const override { return size_; }

  // Implement IDraggableObjectsView.
  maybe<int> entity() const override {
    return static_cast<int>( e_colview_entity::units_at_gate );
  }

  ui::View& view() noexcept override { return *this; }
  ui::View const& view() const noexcept override {
    return *this;
  }

  void draw( rr::Renderer& renderer,
             Coord coord ) const override {
    rr::Painter painter = renderer.painter();
    painter.draw_empty_rect( bounds( coord ).with_inc_size(),
                             rr::Painter::e_border_mode::inside,
                             BROWN_COLOR );
    for( auto [unit_id, unit_pos] : positioned_units_ ) {
      if( dragging_ == unit_id ) continue;
      Coord draw_pos   = unit_pos.as_if_origin_were( coord );
      Unit const& unit = ss_.units.unit_for( unit_id );
      UnitFlagRenderInfo const flag_info =
          euro_unit_flag_render_info( unit, /*viewer=*/nothing,
                                      UnitFlagOptions{} );
      render_unit(
          renderer, draw_pos, unit,
          UnitRenderOptions{
            .flag   = flag_info,
            .shadow = UnitShadow{
              .color = config_colony.colors
                           .unit_shadow_color_light } } );
      if( selected_ == unit_id )
        rr::draw_empty_rect_faded_corners(
            renderer,
            rect{ .origin = draw_pos, .size = g_tile_delta } -
                size{ .w = 1, .h = 1 },
            pixel::green() );
    }
  }

  // Implement AwaitView.
  wait<> perform_click(
      input::mouse_button_event_t const& event ) override {
    if( event.buttons != input::e_mouse_button_event::left_up )
      co_return;
    CHECK( event.pos.is_inside( bounds( {} ) ) );
    for( auto [unit_id, unit_pos] : positioned_units_ ) {
      if( event.pos.is_inside(
              Rect::from( unit_pos, g_tile_delta ) ) ) {
        co_await click_on_unit( unit_id );
      }
    }
  }

  maybe<UnitId> contains_unit( Coord const& where ) const {
    for( PositionedUnit const& pu : positioned_units_ )
      if( where.is_inside( Rect::from( pu.pos, g_tile_delta ) ) )
        return pu.id;
    return nothing;
  }

  maybe<DraggableObjectWithBounds<ColViewObject>> object_here(
      Coord const& where ) const override {
    for( PositionedUnit const& pu : positioned_units_ ) {
      auto rect = Rect::from( pu.pos, g_tile_delta );
      if( where.is_inside( rect ) )
        return DraggableObjectWithBounds<ColViewObject>{
          .obj    = ColViewObject::unit{ .id = pu.id },
          .bounds = rect };
    }
    return nothing;
  }

  maybe<CanReceiveDraggable<ColViewObject>> can_receive_unit(
      UnitId dragged, e_colview_entity /*from*/,
      Coord const& where ) const {
    auto& unit = ss_.units.unit_for( dragged );
    // Player should not be dragging ships or wagons.
    CHECK( unit.desc().cargo_slots == 0 );
    // See if the draga target is over top of a unit.
    maybe<UnitId> over_unit_id = contains_unit( where );
    if( !over_unit_id ) {
      // The player is moving a unit outside of the colony, let's
      // check if the unit is already outside the colony, in
      // which case there is no reason to drag the unit here.
      if( is_unit_on_map( ss_.units, dragged ) ) return nothing;
      // The player is moving the unit outside the colony, which
      // is always allowed, at least for now. If the unit is in
      // the colony (as opposed to cargo) and there is a stockade
      // then we won't allow the population to be reduced below
      // three, but that will be checked in the confirmation
      // stage.
      return CanReceiveDraggable<ColViewObject>::yes{
        .draggable = ColViewObject::unit{ .id = dragged } };
    }
    Unit const& target_unit =
        ss_.units.unit_for( *over_unit_id );
    if( target_unit.desc().cargo_slots == 0 ) return nothing;
    // This is basically to prevent wagon trains from receiving
    // units as cargo.
    if( !target_unit.desc().can_hold_unit_cargo )
      // It may not be obvious to the player why this is being
      // rejected, so give this one a message.
      return CanReceiveDraggable<ColViewObject>::no_with_msg{
        .msg = fmt::format( "[{}] cannot hold units as cargo.",
                            target_unit.desc().name_plural ) };
    // Check if the target_unit is already holding the dragged
    // unit.
    maybe<UnitId> maybe_holder_of_dragged =
        is_unit_onboard( ss_.units, dragged );
    if( maybe_holder_of_dragged &&
        *maybe_holder_of_dragged == over_unit_id )
      // The dragged unit is already in the cargo of the target
      // unit.
      return nothing;
    // At this point, the unit is being dragged on top of another
    // unit that has cargo slots but is not already being held by
    // that unit, so we need to check if the unit fits.
    if( !target_unit.cargo().fits_somewhere(
            ss_.units, Cargo::unit{ dragged } ) )
      return nothing;
    return CanReceiveDraggable<ColViewObject>::yes{
      .draggable = ColViewObject::unit{ .id = dragged } };
  }

  // Implement IDragSinkUserEdit.
  wait<maybe<ColViewObject>> user_edit_object(
      ColViewObject const& o, int from_entity,
      Coord const ) const override {
    CONVERT_ENTITY( from_enum, from_entity );
    switch( from_enum ) {
      case e_colview_entity::units_at_gate:
      case e_colview_entity::commodities:
      case e_colview_entity::cargo: //
        co_return o;
      case e_colview_entity::land:
      case e_colview_entity::buildings: {
        // We're dragging an in-colony unit to the gate, so check
        // if it is the last remaining colonist in the colony
        // and, if so, if the user wants to equip it with any
        // horses, tools, or muskets as it leaves and the colony
        // disappears.
        UNWRAP_CHECK( draggable_unit,
                      o.get_if<ColViewObject::unit>() );
        ColViewObject::unit new_draggable_unit = draggable_unit;
        Unit const& unit =
            ss_.units.unit_for( draggable_unit.id );
        maybe<ColonyEquipOption> const equip_options =
            co_await ask_transorm_unit_on_leave( ts_, colony_,
                                                 unit );
        if( equip_options.has_value() )
          new_draggable_unit.transformed = *equip_options;
        co_return new_draggable_unit;
      }
      case e_colview_entity::population:
      case e_colview_entity::title_bar:
      case e_colview_entity::production:
        FATAL( "unexpected source entity." );
    }
  }

  wait<base::valid_or<DragRejection>> sink_check(
      ColViewObject const&, int from_entity,
      Coord const ) override {
    CONVERT_ENTITY( from_enum, from_entity );
    switch( from_enum ) {
      case e_colview_entity::units_at_gate:
      case e_colview_entity::commodities:
      case e_colview_entity::cargo: //
        co_return base::valid;
      case e_colview_entity::land:
      case e_colview_entity::buildings: //
        if( check_stockade_3( colony_ ) )
          co_return DragRejection{
            .reason = string( kReduceStockadeThreeMsg ) };
        if( co_await check_abandon( colony_, ts_.gui ) )
          // If we're rejecting then that means that the player
          // has opted not to abandon the colony, so there is no
          // need to display a reason message.
          co_return DragRejection{ .reason = nothing };
        if( auto msg = check_seige(); msg.has_value() )
          co_return DragRejection{ .reason = *msg };
        co_return base::valid;
      case e_colview_entity::population:
      case e_colview_entity::title_bar:
      case e_colview_entity::production:
        FATAL( "unexpected source entity." );
    }
  }

  maybe<CanReceiveDraggable<ColViewObject>>
  can_cargo_unit_receive_commodity(
      Commodity const& comm, e_colview_entity from,
      UnitId cargo_unit_id ) const {
    Unit const& target_unit =
        ss_.units.unit_for( cargo_unit_id );
    CHECK( target_unit.desc().cargo_slots != 0 );
    // Check if the target_unit is already holding the dragged
    // commodity.
    if( from == e_colview_entity::cargo ) {
      CHECK( selected_.has_value() );
      CHECK(
          ss_.units.unit_for( *selected_ ).desc().cargo_slots >
          0 );
      if( cargo_unit_id == *selected_ )
        // The commodity is already in the cargo of the unit
        // under the mouse.
        return nothing;
    }
    // At this point, the commodity is being dragged on top of a
    // unit that has cargo slots but is not already being held by
    // that unit, so we need to check if the commodity fits.
    int max_q =
        target_unit.cargo().max_commodity_quantity_that_fits(
            comm.type );
    if( max_q == 0 ) return nothing;
    // We may need to adjust the quantity.
    Commodity new_comm = comm;
    new_comm.quantity  = std::min( new_comm.quantity, max_q );
    CHECK( new_comm.quantity > 0 );
    return CanReceiveDraggable<ColViewObject>::yes{
      .draggable =
          ColViewObject::commodity{ .comm = new_comm } };
  }

  static maybe<UnitTransformationFromCommodity>
  transformed_unit_composition_from_commodity(
      Unit const& unit, Commodity const& comm ) {
    vector<UnitTransformationFromCommodity> possibilities =
        with_commodity_added( unit, comm );
    adjust_for_independence_status(
        possibilities,
        // FIXME
        /*independence_declared=*/false );

    erase_if( possibilities, []( auto const& xform_res ) {
      for( auto [mod, delta] : xform_res.modifier_deltas ) {
        if( delta == e_unit_type_modifier_delta::none ) continue;
        if( !config_unit_type.composition.modifier_traits[mod]
                 .player_can_grant )
          return true;
      }
      return false; // don't erase.
    } );

    maybe<UnitTransformationFromCommodity> res;
    if( possibilities.size() == 1 ) res = possibilities[0];
    return res;
  }

  maybe<CanReceiveDraggable<ColViewObject>>
  can_unit_receive_commodity( Commodity const& comm,
                              e_colview_entity /*from*/,
                              UnitId id ) const {
    // We are dragging a commodity over a unit that does not have
    // a cargo hold. This could be valid if we are e.g. giving
    // muskets to a colonist.
    UNWRAP_RETURN( xform_res,
                   transformed_unit_composition_from_commodity(
                       ss_.units.unit_for( id ), comm ) );
    return CanReceiveDraggable<ColViewObject>::yes{
      .draggable = ColViewObject::commodity{
        .comm =
            with_quantity( comm, xform_res.quantity_used ) } };
  }

  maybe<CanReceiveDraggable<ColViewObject>>
  can_receive_commodity( Commodity const& comm,
                         e_colview_entity from,
                         Coord const& where ) const {
    maybe<UnitId> over_unit_id = contains_unit( where );
    if( !over_unit_id ) return nothing;
    Unit const& target_unit =
        ss_.units.unit_for( *over_unit_id );
    if( target_unit.desc().cargo_slots != 0 )
      return can_cargo_unit_receive_commodity( comm, from,
                                               *over_unit_id );
    else
      return can_unit_receive_commodity( comm, from,
                                         *over_unit_id );
  }

  maybe<CanReceiveDraggable<ColViewObject>> can_receive(
      ColViewObject const& o, int from_entity,
      Coord const& where ) const override {
    CONVERT_ENTITY( from_enum, from_entity );
    CHECK( where.is_inside( bounds( {} ) ) );
    if( !where.is_inside( bounds( {} ) ) ) return nothing;
    return overload_visit(
        o, //
        [&]( ColViewObject::unit const& unit ) {
          return can_receive_unit( unit.id, from_enum, where );
        },
        [&]( ColViewObject::commodity const& comm ) {
          return can_receive_commodity( comm.comm, from_enum,
                                        where );
        } );
  }

  wait<> drop( ColViewObject const& o,
               Coord const& where ) override {
    maybe<UnitId> target_unit_id = contains_unit( where );
    if( target_unit_id.has_value() ) {
      Unit& target_unit = ss_.units.unit_for( *target_unit_id );
      // We're dragging something onto/into a unit, so clear or-
      // ders and subtract some movement points depending on the
      // unit type.
      target_unit.clear_orders();
      adjust_mv_points_from_drag( target_unit );
    }
    overload_visit(
        o, //
        [&]( ColViewObject::unit const& draggable_unit ) {
          if( draggable_unit.transformed.has_value() ) {
            Unit& to_transform =
                ss_.units.unit_for( draggable_unit.id );
            // This will change the unit type and modify colony
            // commodity quantities.
            perform_colony_equip_option(
                ss_, ts_, colony_, to_transform,
                *draggable_unit.transformed );
          }
          if( target_unit_id ) {
            UnitOwnershipChanger( ss_, draggable_unit.id )
                .change_to_cargo( *target_unit_id,
                                  /*starting_slot=*/0 );
            // !! Need to fall through here since we may have
            // abandoned the colony.
          } else {
            UnitOwnershipChanger( ss_, draggable_unit.id )
                .change_to_map_non_interactive(
                    ts_, colony_.location );
            // This is not strictly necessary, but as a conve-
            // nience to the user, clear the orders, otherwise it
            // would be sentry'd, which is probably not what the
            // player wants.
            ss_.units.unit_for( draggable_unit.id )
                .clear_orders();
          }
          // Check if we've abandoned the colony.
          if( colony_population( colony_ ) == 0 )
            throw colony_abandon_interrupt{};
        },
        [&]( ColViewObject::commodity const& comm ) {
          CHECK( target_unit_id );
          Unit& target_unit =
              ss_.units.unit_for( *target_unit_id );
          if( target_unit.desc().cargo_slots > 0 ) {
            add_commodity_to_cargo(
                ss_.units, comm.comm,
                ss_.units.unit_for( *target_unit_id ).cargo(),
                /*slot=*/0,
                /*try_other_slots=*/true );
          } else {
            // We are dragging a commodity over a unit that does
            // not have a cargo hold. This could be valid if we
            // are e.g. giving muskets to a colonist.
            Commodity const& dropping_comm = comm.comm;
            UNWRAP_CHECK(
                xform_res,
                transformed_unit_composition_from_commodity(
                    target_unit, dropping_comm ) );
            CHECK( xform_res.quantity_used ==
                   dropping_comm.quantity );
            change_unit_type( ss_, ts_, target_unit,
                              xform_res.new_comp );
            CHECK( ss_.units.coord_for( target_unit.id() ) ==
                   colony_.location );
            // The unit, being at the colony gate, is actually on
            // the map at the site of this colony. In the event
            // that we are e.g. changing a colonist to a scout
            // (whsch has a sighting radius of two) we should
            // call this function to update the rendered map
            // along with anything else that needs to be done.
            UnitOwnershipChanger( ss_, target_unit.id() )
                .change_to_map_non_interactive(
                    ts_, colony_.location );
            // Note that we clear orders and deal with movement
            // points for the target unit at the top of this
            // function.
          }
        } );
    co_return;
  }

  bool try_drag( ColViewObject const& o,
                 Coord const& /*where*/ ) override {
    UNWRAP_CHECK( draggable_unit,
                  o.get_if<ColViewObject::unit>() );
    bool is_cargo_unit = ss_.units.unit_for( draggable_unit.id )
                             .desc()
                             .cargo_slots > 0;
    if( is_cargo_unit ) return false;
    dragging_ = draggable_unit.id;
    return true;
  }

  void cancel_drag() override { dragging_ = nothing; }

  wait<> disown_dragged_object() override {
    UNWRAP_CHECK( unit_id, dragging_ );
    UnitOwnershipChanger( ss_, unit_id ).change_to_free();
    co_return;
  }

 private:
  void set_selected_unit( maybe<UnitId> id ) {
    selected_ = id;
    cargo_view_->set_unit( id );
  }

  wait<> click_on_unit( UnitId id ) {
    lg.info( "clicked on unit {}.",
             debug_string( ss_.units, id ) );
    Unit& unit = ss_.units.unit_for( id );
    if( selected_ != id ) {
      set_selected_unit( id );
      // The first time we select a unit, just select it, but
      // don't pop up the orders menu until the second click.
      // This should make a more polished feel for the UI, and
      // also allow viewing a ship's cargo without popping up the
      // orders menu.
      co_return;
    }
    if( auto damaged =
            unit.orders().get_if<unit_orders::damaged>();
        damaged.has_value() ) {
      co_await ts_.gui.message_box( ship_still_damaged_message(
          damaged->turns_until_repair ) );
      co_return;
    }
    // FIXME: need to replace the two below calls with a more
    // robust (non-string-based) approach.
    ChoiceConfig config{
      .msg     = "What would you like to do?",
      .options = {
        { .key = "orders", .display_name = "Change Orders" },
      } };
    if( unit.cargo().slots_total() == 0 )
      // If we try to strip a unit that can carry cargo then we
      // might crash.
      config.options.push_back(
          { .key = "strip", .display_name = "Strip Unit" } );
    if( can_bless_missionaries( colony_ ) &&
        unit_can_be_blessed( unit.type_obj() ) )
      config.options.push_back(
          { .key          = "missionary",
            .display_name = "Bless as Missionary" } );
    maybe<string> const mode =
        co_await ts_.gui.optional_choice( config );
    if( mode == "orders" ) {
      ChoiceConfig config{
        .msg     = "Change unit orders to:",
        .options = {
          { .key = "clear", .display_name = "Clear Orders" },
          { .key = "sentry", .display_name = "Sentry" },
          { .key = "fortify", .display_name = "Fortify" } } };
      maybe<string> const new_orders =
          co_await ts_.gui.optional_choice( config );
      if( new_orders == "clear" )
        unit.clear_orders();
      else if( new_orders == "sentry" )
        unit.sentry();
      else if( new_orders == "fortify" ) {
        if( !unit.orders().holds<unit_orders::fortified>() )
          // This will place them in the "fortifying" state and
          // will consume all movement points. The only time we
          // don't want to do this is if the unit's state is
          // "fortified", since then this would represent a pes-
          // simisation.
          unit.start_fortify();
      }
    } else if( mode == "strip" ) {
      UnitComposition const old_comp = unit.composition();
      strip_unit_to_base_type( ss_, ts_, unit, colony_ );
      if( unit.composition() != old_comp ) {
        // If the unit has changed in any way then 1) clear or-
        // ders (this is especially important for a pioneer that
        // is in e.g. a "plow" state when its tools are stripped
        // so that it doesn't end up in an inconsistent state),
        // and 2) adjust its movement points.
        unit.clear_orders();
        adjust_mv_points_from_drag( unit );
      }
    } else if( mode == "missionary" ) {
      // TODO: play blessing tune.
      bless_as_missionary( ss_, ts_, colony_, unit );
    }
  }

  void sort_by_ordering( vector<UnitId>& sort_me ) const {
    sort( sort_me.begin(), sort_me.end(),
          [&]( UnitId const lhs, UnitId const rhs ) {
            // Reverse sorting since later ordered units are
            // always considered at the "front" in the colony.
            return ss_.units.unit_ordering( rhs ) <
                   ss_.units.unit_ordering( lhs );
          } );
  }

  void update_this_and_children() override {
    auto const& colony = ss_.colonies.colony_for( colony_.id );
    auto const units   = [&] {
      auto const& units_set =
          ss_.units.from_coord( colony.location );
      vector<UnitId> res;
      res.reserve( units_set.size() );
      for( GenericUnitId const generic_id : units_set )
        res.push_back( ss_.units.check_euro_unit( generic_id ) );
      sort_by_ordering( res );
      return res;
    }();
    auto unit_pos = Coord{} + Delta{ .w = 1, .h = 16 };
    positioned_units_.clear();
    maybe<UnitId> first_with_cargo;
    for( GenericUnitId generic_id : units ) {
      UnitId const unit_id =
          ss_.units.check_euro_unit( generic_id );
      positioned_units_.push_back(
          { .id = unit_id, .pos = unit_pos } );
      unit_pos.x += 32;
      if( !first_with_cargo.has_value() &&
          ss_.units.unit_for( unit_id ).desc().cargo_slots > 0 )
        first_with_cargo = unit_id;
    }
    if( selected_.has_value() &&
        find( units.begin(), units.end(), *selected_ ) ==
            units.end() )
      set_selected_unit( nothing );
    if( !selected_.has_value() )
      set_selected_unit( first_with_cargo );
  }

  struct PositionedUnit {
    UnitId id;
    Coord pos; // relative to upper left of this CargoView.
  };

  vector<PositionedUnit> positioned_units_;
  // FIXME: this gets reset whenever we recomposite. We need to
  // either put this in a global place, or not recreate all of
  // these view objects each time we recomposite (i.e., reuse
  // them).
  maybe<UnitId> selected_;

  CargoView* cargo_view_;
  Delta size_;
  maybe<UnitId> dragging_;
};

/****************************************************************
** Compositing
*****************************************************************/
struct CompositeColSubView : public ui::InvisibleView,
                             public ColonySubView {
  CompositeColSubView(
      IEngine& engine, SS& ss, TS& ts, Player& player,
      Colony& colony, Delta size,
      std::vector<ui::OwningPositionedView> views )
    : ui::InvisibleView( size, std::move( views ) ),
      ColonySubView( engine, ss, ts, player, colony ) {
    for( ui::PositionedView v : *this ) {
      auto* col_view = dynamic_cast<ColonySubView*>( v.view );
      CHECK( col_view );
      ptrs_.push_back( col_view );
    }
    CHECK( int( ptrs_.size() ) == count() );
  }

  ui::View& view() noexcept override { return *this; }
  ui::View const& view() const noexcept override {
    return *this;
  }

  // Implement AwaitView.
  wait<> perform_click(
      input::mouse_button_event_t const& event ) override {
    for( int i = 0; i < count(); ++i ) {
      ui::PositionedView pos_view = at( i );
      if( !event.pos.is_inside( pos_view.rect() ) ) continue;
      input::event_t const shifted_event =
          input::mouse_origin_moved_by(
              event, pos_view.coord.distance_from_origin() );
      UNWRAP_CHECK(
          shifted_mouse_button_event,
          shifted_event.get_if<input::mouse_button_event_t>() );
      // Need to co_await so that shifted_event stays alive.
      co_await ptrs_[i]->perform_click(
          shifted_mouse_button_event );
      break;
    }
  }

  maybe<PositionedDraggableSubView<ColViewObject>> view_here(
      Coord coord ) override {
    for( int i = 0; i < count(); ++i ) {
      ui::PositionedView pos_view = at( i );
      if( !coord.is_inside( pos_view.rect() ) ) continue;
      maybe<PositionedDraggableSubView<ColViewObject>> p_view =
          ptrs_[i]->view_here(
              coord.with_new_origin( pos_view.coord ) );
      if( !p_view ) continue;
      p_view->upper_left =
          p_view->upper_left.as_if_origin_were( pos_view.coord );
      return p_view;
    }
    if( coord.is_inside( bounds( {} ) ) )
      return PositionedDraggableSubView<ColViewObject>{
        this, Coord{} };
    return nothing;
  }

  // Implement ColonySubView.
  maybe<DraggableObjectWithBounds<ColViewObject>> object_here(
      Coord const& coord ) const override {
    for( int i = 0; i < count(); ++i ) {
      ui::PositionedViewConst pos_view = at( i );
      if( !coord.is_inside( pos_view.rect() ) ) continue;
      maybe<DraggableObjectWithBounds<ColViewObject>> obj =
          ptrs_[i]->object_here(
              coord.with_new_origin( pos_view.coord ) );
      if( !obj ) continue;
      obj->bounds =
          obj->bounds.as_if_origin_were( pos_view.coord );
      return obj;
    }
    // This view itself has no objects.
    return nothing;
  }

  // Implement IDraggableObjectsView.
  maybe<int> entity() const override { return nothing; }

  void update_this_and_children() override {
    for( ColonySubView* p : ptrs_ )
      p->update_this_and_children();
  }

  vector<ColonySubView*> ptrs_;
};

void recomposite( IEngine& engine, SS& ss, TS& ts,
                  Player& player, Colony& colony,
                  Delta const& canvas_size ) {
  lg.trace( "recompositing colony view." );
  g_composition.id          = colony.id;
  g_composition.canvas_size = canvas_size;

  g_composition.top_level = nullptr;
  g_composition.entities.clear();
  vector<ui::OwningPositionedView> views;

  Coord pos;
  Delta available;

  // [Title Bar] ------------------------------------------------
  auto title_bar =
      TitleBar::create( engine, ss, ts, player, colony,
                        Delta{ .w = canvas_size.w, .h = 10 } );
  g_composition.entities[e_colview_entity::title_bar] =
      title_bar.get();
  pos = Coord{};
  Y const title_bar_bottom =
      title_bar->bounds( pos ).bottom_edge();
  views.push_back( ui::OwningPositionedView{
    .view = std::move( title_bar ), .coord = pos } );

  // [MarketCommodities] ----------------------------------------
  W comm_block_width =
      canvas_size.w / SX{ refl::enum_count<e_commodity> };
  comm_block_width =
      std::clamp( comm_block_width, kCommodityTileSize.w, 32 );
  auto market_commodities = MarketCommodities::create(
      engine, ss, ts, player, colony, comm_block_width );
  g_composition.entities[e_colview_entity::commodities] =
      market_commodities.get();
  pos = Coord::from_gfx( gfx::centered_at_bottom(
      market_commodities->delta(),
      Rect::from( Coord{}, canvas_size ) ) );
  auto const market_commodities_top = pos.y;
  views.push_back( ui::OwningPositionedView{
    .view = std::move( market_commodities ), .coord = pos } );

  // [Middle Strip] ---------------------------------------------
  Delta middle_strip_size{ canvas_size.w, 32 + 32 + 16 };
  Y const middle_strip_top =
      market_commodities_top - middle_strip_size.h;

  // [Population] -----------------------------------------------
  auto population_view = PopulationView::create(
      engine, ss, ts, player, colony,
      middle_strip_size.with_width( middle_strip_size.w / 3 ) );
  g_composition.entities[e_colview_entity::population] =
      population_view.get();
  pos = Coord{ .x = 0, .y = middle_strip_top };
  X const population_right_edge =
      population_view->bounds( pos ).right_edge();
  views.push_back( ui::OwningPositionedView{
    .view = std::move( population_view ), .coord = pos } );

  // [Cargo] ----------------------------------------------------
  auto cargo_view = CargoView::create(
      engine, ss, ts, player, colony,
      middle_strip_size.with_width( middle_strip_size.w / 3 )
          .with_height( 32 ) );
  g_composition.entities[e_colview_entity::cargo] =
      cargo_view.get();
  pos = Coord{ .x = population_right_edge,
               .y = middle_strip_top + 32 + 16 };
  X const cargo_right_edge =
      cargo_view->bounds( pos ).right_edge();
  auto* p_cargo_view = cargo_view.get();
  views.push_back( ui::OwningPositionedView{
    .view = std::move( cargo_view ), .coord = pos } );

  // [Units at Gate outside colony] -----------------------------
  auto units_at_gate_view = UnitsAtGateColonyView::create(
      engine, ss, ts, player, colony, p_cargo_view,
      middle_strip_size.with_width( middle_strip_size.w / 3 )
          .with_height( middle_strip_size.h - 32 ) );
  g_composition.entities[e_colview_entity::units_at_gate] =
      units_at_gate_view.get();
  pos =
      Coord{ .x = population_right_edge, .y = middle_strip_top };
  views.push_back( ui::OwningPositionedView{
    .view = std::move( units_at_gate_view ), .coord = pos } );

  // [Production] -----------------------------------------------
  auto production_view = ProductionView::create(
      engine, ss, ts, player, colony,
      middle_strip_size.with_width( middle_strip_size.w / 3 ) );
  g_composition.entities[e_colview_entity::production] =
      production_view.get();
  pos = Coord{ .x = cargo_right_edge, .y = middle_strip_top };
  views.push_back( ui::OwningPositionedView{
    .view = std::move( production_view ), .coord = pos } );

  // [ColonyLandView] -------------------------------------------
  available = Delta{ canvas_size.w,
                     middle_strip_top - title_bar_bottom };

  H max_landview_height = available.h;

  ColonyLandView::e_render_mode land_view_mode =
      ColonyLandView::e_render_mode::_6x6;
  if( ColonyLandView::size_needed( land_view_mode ).h >
      max_landview_height )
    land_view_mode = ColonyLandView::e_render_mode::_5x5;
  if( ColonyLandView::size_needed( land_view_mode ).h >
      max_landview_height )
    land_view_mode = ColonyLandView::e_render_mode::_3x3;
  auto land_view = ColonyLandView::create(
      engine, ss, ts, player, colony, land_view_mode );
  g_composition.entities[e_colview_entity::land] =
      land_view.get();
  pos = g_composition.entities[e_colview_entity::title_bar]
            ->view()
            .bounds( Coord{} )
            .lower_right() -
        Delta{ .w = land_view->delta().w };
  X const land_view_left_edge = pos.x;
  views.push_back( ui::OwningPositionedView{
    .view = std::move( land_view ), .coord = pos } );

  // [Buildings] ------------------------------------------------
  Delta buildings_size{
    .w = land_view_left_edge - 0,
    .h = middle_strip_top - title_bar_bottom };
  auto buildings = ColViewBuildings::create(
      engine, ss, ts, player, colony, buildings_size );
  g_composition.entities[e_colview_entity::buildings] =
      buildings.get();
  pos = Coord{ .x = 0, .y = title_bar_bottom };
  views.push_back( ui::OwningPositionedView{
    .view = std::move( buildings ), .coord = pos } );

  // [Finish] ---------------------------------------------------
  auto invisible_view = std::make_unique<CompositeColSubView>(
      engine, ss, ts, player, colony, canvas_size,
      std::move( views ) );
  invisible_view->set_delta( canvas_size );
  g_composition.top_level = std::move( invisible_view );

  for( auto e : refl::enum_values<e_colview_entity> ) {
    CHECK( g_composition.entities.contains( e ),
           "colview entity {} is missing.", e );
  }
}

} // namespace

/****************************************************************
** Public API
*****************************************************************/
ColonySubView& colview_top_level() {
  CHECK( g_composition.top_level );
  return *g_composition.top_level;
}

// FIXME: a lot of this needs to be de-duped with the corre-
// sponding code in old-world-view.
void colview_drag_n_drop_draw(
    SS& ss, rr::Renderer& renderer,
    DragState<ColViewObject> const& state,
    Coord const& canvas_origin ) {
  Coord sprite_upper_left = state.where - state.click_offset +
                            canvas_origin.distance_from_origin();
  // Render the dragged item.
  overload_visit(
      state.object,
      [&]( ColViewObject::unit const& o ) {
        render_unit( renderer, sprite_upper_left,
                     ss.units.unit_for( o.id ),
                     UnitRenderOptions{} );
      },
      [&]( ColViewObject::commodity const& o ) {
        render_commodity_16( renderer, sprite_upper_left,
                             o.comm.type );
      } );
  // Render any indicators on top of it.
  switch( state.indicator ) {
    using e = e_drag_status_indicator;
    case e::none:
      break;
    case e::bad: {
      rr::Typer typer =
          renderer.typer( sprite_upper_left, gfx::pixel::red() );
      typer.write( "X" );
      break;
    }
    case e::good: {
      rr::Typer typer = renderer.typer( sprite_upper_left,
                                        gfx::pixel::green() );
      typer.write( "+" );
      if( state.source_requests_edit ) {
        auto mod_pos = state.where;
        mod_pos.y -= H{ typer.dimensions_for_line( "?" ).h };
        mod_pos -= state.click_offset;
        auto typer_mod =
            renderer.typer( mod_pos, gfx::pixel::green() );
        typer_mod.write( "?" );
      }
      break;
    }
  }
}

ColonyProduction const& colview_production() {
  return g_production;
}

void update_colony_view( SSConst const& ss,
                         Colony const& colony ) {
  update_production( ss, colony );
  ColonySubView& top = colview_top_level();
  top.update_this_and_children();
}

void update_production( SSConst const& ss,
                        Colony const& colony ) {
  g_production = production_for_colony( ss, colony );
}

void set_colview_colony( IEngine& engine, SS& ss, TS& ts,
                         Player& player, Colony& colony ) {
  update_production( ss, colony );
  // TODO: compute squares around this colony that are being
  // worked by other colonies.
  auto const r = main_window_logical_rect(
      engine.video(), engine.window(), engine.resolutions() );
  recomposite( engine, ss, ts, player, colony, r.size );
}

} // namespace rn
