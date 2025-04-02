// Repository: objective-audio/ui
// File: Sources/ui/include/ui/metal/view/yas_ui_metal_view_controller.mm

//
//  yas_ui_metal_view_controller.mm
//

#include "yas_ui_metal_view_controller.h"
#include "yas_ui_metal_view.h"
#include "yas_ui_metal_view_controller_dependency.h"
#include "yas_ui_metal_view_controller_dependency_objc.h"

#include <objc-utils/unowned.h>
#include <ui/background/yas_ui_background.h>
#include <ui/color/yas_ui_rgb_color.h>
#include <ui/metal/view/yas_ui_metal_view_utils.h>
#include <ui/metal/yas_ui_metal_system.h>
#include <ui/view_look/yas_ui_view_look.h>
#include <observing/umbrella.hpp>

NS_ASSUME_NONNULL_BEGIN

using namespace yas;
using namespace yas::ui;

namespace yas::ui {
struct metal_view_cpp {
    std::shared_ptr<view_look> const view_look = ui::view_look::make_shared();
    std::shared_ptr<renderer_for_view> renderer{nullptr};
    int appearance_updating_delay = 0;
    observing::canceller_pool bg_pool;
};
}  // namespace yas::ui

@interface YASUIMetalViewController () <MTKViewDelegate, YASUIMetalViewDelegate>
#if TARGET_OS_IPHONE
@property (nonatomic) id<UITraitChangeRegistration> userInterfaceStyleChangeRegistration;
#endif
@end

@implementation YASUIMetalViewController {
    ui::metal_view_cpp _cpp;
}

#if TARGET_OS_IPHONE
- (instancetype)initWithNibName:(nullable NSString *)nibNameOrNil bundle:(nullable NSBundle *)nibBundleOrNil {
#elif TARGET_OS_MAC
- (instancetype)initWithNibName:(nullable NSNibName)nibNameOrNil bundle:(nullable NSBundle *)nibBundleOrNil {
#endif
    self = [super initWithNibName:nibNameOrNil bundle:nibBundleOrNil];
    if (self) {
        [self initCommon];
    }
    return self;
}

- (nullable instancetype)initWithCoder:(NSCoder *)coder {
    self = [super initWithCoder:coder];
    if (self) {
        [self initCommon];
    }
    return self;
}

- (void)initCommon {
}

#if TARGET_OS_IPHONE
- (void)dealloc {
    [self.view unregisterForTraitChanges:self.userInterfaceStyleChangeRegistration];
}
#endif

- (void)loadView {
    if (self.nibName || self.nibBundle) {
        [super loadView];
    } else {
        auto view = objc_ptr_with_move_object([[YASUIMetalView alloc] initWithFrame:CGRectMake(0, 0, 256, 256)
                                                                             device:nil]);
        self.view = view.object();
    }
}

- (void)viewDidLoad {
    [super viewDidLoad];

#if (!TARGET_OS_IPHONE && TARGET_OS_MAC)
    [self.metalView addObserver:self
                     forKeyPath:@"effectiveAppearance"
                        options:NSKeyValueObservingOptionNew
                        context:nil];
#endif

    self.metalView.delegate = self;
    self.metalView.uiDelegate = self;

    [self updateViewLookSizesWithDrawableSize:self.metalView.drawableSize];
    self->_cpp.view_look->set_appearance(self.metalView.uiAppearance);

#if TARGET_OS_IPHONE
    self.userInterfaceStyleChangeRegistration =
        [self.view registerForTraitChanges:@[UITraitUserInterfaceStyle.self]
                               withHandler:[self](id<UITraitEnvironment> traitEnvironment,
                                                  UITraitCollection *previousCollection) {
                                   [self appearanceDidChange:self.uiAppearance];
                               }];
#endif
}

#if (!TARGET_OS_IPHONE && TARGET_OS_MAC)
- (void)dealloc {
    [self.metalView removeObserver:self forKeyPath:@"effectiveAppearance"];

    yas_super_dealloc();
}

- (void)observeValueForKeyPath:(nullable NSString *)keyPath
                      ofObject:(nullable id)object
                        change:(nullable NSDictionary<NSKeyValueChangeKey, id> *)change
                       context:(nullable void *)context {
    if ([keyPath isEqualToString:@"effectiveAppearance"]) {
        [self appearanceDidChange:self.metalView.uiAppearance];
    }
}

- (ui::appearance)uiAppearance {
    return self.metalView.uiAppearance;
}
#endif

#if TARGET_OS_IPHONE
- (ui::appearance)uiAppearance {
    switch (self.traitCollection.userInterfaceStyle) {
        case UIUserInterfaceStyleDark:
            return ui::appearance::dark;
        default:
            return ui::appearance::normal;
    }
}
#endif

- (void)appearanceDidChange:(yas::ui::appearance)appearance {
    self->_cpp.appearance_updating_delay = 2;
}

- (YASUIMetalView *)metalView {
    return (YASUIMetalView *)self.view;
}

- (std::shared_ptr<yas::ui::view_look> const &)view_look {
    return self->_cpp.view_look;
}

- (void)configure_with_metal_system:(std::shared_ptr<yas::ui::metal_system_for_view> const &)metal_system
                           renderer:(std::shared_ptr<yas::ui::renderer_for_view> const &)renderer
                      event_manager:(std::shared_ptr<yas::ui::event_manager_for_view> const &)event_manager {
    if (metal_system) {
        self.metalView.device = metal_system->mtlDevice();
        self.metalView.sampleCount = metal_system->sample_count();
    } else {
        self.metalView.device = nil;
        self.metalView.sampleCount = 1;
    }

    self->_cpp.renderer = renderer;

    renderer
        ->observe_background_color([self](ui::color const &color) {
            self.metalView.clearColor = MTLClearColorMake(color.red, color.green, color.blue, color.alpha);
        })
        .end()
        ->add_to(self->_cpp.bg_pool);

    [self.metalView configure];
    [self.metalView set_event_manager:event_manager];
}

- (std::shared_ptr<yas::ui::renderer_for_view> const &)renderer {
    return self->_cpp.renderer;
}

#pragma mark -

- (void)setPaused:(BOOL)pause {
    self.metalView.paused = pause;
}

- (BOOL)isPaused {
    return self.metalView.isPaused;
}

#pragma mark - MTKViewDelegate

- (void)mtkView:(YASUIMetalView *)view drawableSizeWillChange:(CGSize)drawable_size {
    [self updateViewLookSizesWithDrawableSize:drawable_size];
}

- (void)drawInMTKView:(YASUIMetalView *)view {
    if (self->_cpp.renderer) {
        self->_cpp.renderer->view_render();
    }

    if (self->_cpp.appearance_updating_delay > 0) {
        self->_cpp.appearance_updating_delay--;

        if (self->_cpp.appearance_updating_delay == 0) {
            self->_cpp.view_look->set_appearance(self.uiAppearance);
        }
    }
}

#pragma mark - YASUIMetalViewDelegate

- (void)uiMetalView:(YASUIMetalView *)view safeAreaInsetsDidChange:(ui::region_insets)insets {
    self->_cpp.view_look->set_safe_area_insets(insets);
}

#pragma mark - Private

- (void)updateViewLookSizesWithDrawableSize:(CGSize)drawable_size {
    self->_cpp.view_look->set_view_sizes(metal_view_utils::to_uint_size(self.view.bounds.size),
                                         metal_view_utils::to_uint_size(drawable_size),
                                         self.metalView.uiSafeAreaInsets);
}

@end

NS_ASSUME_NONNULL_END
