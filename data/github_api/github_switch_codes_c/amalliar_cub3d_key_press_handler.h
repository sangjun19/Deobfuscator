/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   key_press_handler.h                                :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: amalliar <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2020/09/06 10:44:40 by amalliar          #+#    #+#             */
/*   Updated: 2020/09/27 12:15:29 by amalliar         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef KEY_PRESS_HANDLER_H
# define KEY_PRESS_HANDLER_H

# include "cub3d.h"
# include "ft_stdlib.h"
# include "ft_string.h"
# include "graphics.h"
# include "keycodes.h"
# include "mlx.h"
# include "snd.h"

int			key_press_handler(int keycode, t_scene *scene);
t_door		*get_door(t_scene *scene, int door_x, int door_y);
t_secret	*get_secret(t_scene *scene, int secr_x, int secr_y);
void		process_interact_request(t_scene *scene, t_player_data *pd, \
				t_map_data *md);
void		switch_weapon(t_player_data *pd, int id, t_scene *scene);

#endif
