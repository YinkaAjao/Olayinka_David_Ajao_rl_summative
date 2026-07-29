import pygame
import numpy as np
import math

class ScoutVisualizer:
    def __init__(self, width=900, height=500):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("RL Agent: Scout Recommendation Engine")
        
        # Fonts
        self.font_title = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_body = pygame.font.SysFont("Arial", 16)
        self.font_large = pygame.font.SysFont("Arial", 36, bold=True)
        
        # Colors
        self.bg_color = (30, 30, 34)
        self.text_color = (240, 240, 240)
        self.scout_color = (70, 130, 180, 150)   # Blueish
        self.athlete_color = (220, 20, 60, 150)  # Reddish
        self.grid_color = (100, 100, 100)
        
        self.trait_labels = ["Speed", "Endurance", "Technical", "Shooting", "Defense", "Physical"]
        
    def render_step(self, info):
        # Pump events so the window doesn't freeze
        pygame.event.pump()
        
        self.screen.fill(self.bg_color)
        
        # 1. Draw Radar Chart (Left Panel)
        self._draw_radar_chart(info["athlete_traits"], info["scout_prefs"], cx=250, cy=270, radius=150)
        
        # 2. Draw Dashboard Info (Right Panel)
        self._draw_dashboard(info, offset_x=550, offset_y=50)
        
        pygame.display.flip()
        
    def _draw_radar_chart(self, athlete_traits, scout_prefs, cx, cy, radius):
        # Title
        title_surf = self.font_title.render("Trait Alignment Radar", True, self.text_color)
        self.screen.blit(title_surf, (cx - title_surf.get_width()//2, cy - radius - 50))
        
        # Draw web grid
        num_axes = len(self.trait_labels)
        angles = [i * (2 * math.pi / num_axes) - (math.pi / 2) for i in range(num_axes)]
        
        for angle in angles:
            end_x = cx + radius * math.cos(angle)
            end_y = cy + radius * math.sin(angle)
            pygame.draw.line(self.screen, self.grid_color, (cx, cy), (end_x, end_y), 1)
            
        # Draw concentric polygons
        for scale in [0.25, 0.5, 0.75, 1.0]:
            points = [
                (cx + radius * scale * math.cos(a), cy + radius * scale * math.sin(a)) 
                for a in angles
            ]
            pygame.draw.polygon(self.screen, self.grid_color, points, 1)
            
        # Labels
        for idx, (angle, label) in enumerate(zip(angles, self.trait_labels)):
            text_surf = self.font_body.render(label, True, self.text_color)
            tx = cx + (radius + 25) * math.cos(angle) - text_surf.get_width()//2
            ty = cy + (radius + 15) * math.sin(angle) - text_surf.get_height()//2
            self.screen.blit(text_surf, (tx, ty))

        # Normalize scout prefs for visual scale (since they sum to 1.0, we scale the max to ~0.8 of radius for visibility)
        max_pref = max(scout_prefs) if max(scout_prefs) > 0 else 1.0
        norm_scout = [p / max_pref * 0.8 for p in scout_prefs]
            
        # Draw Scout Polygon (Blue)
        scout_pts = [(cx + radius * val * math.cos(a), cy + radius * val * math.sin(a)) for val, a in zip(norm_scout, angles)]
        pygame.draw.polygon(self.screen, (70, 130, 180), scout_pts, 3)
        
        # Draw Athlete Polygon (Red)
        athlete_pts = [(cx + radius * val * math.cos(a), cy + radius * val * math.sin(a)) for val, a in zip(athlete_traits, angles)]
        pygame.draw.polygon(self.screen, (220, 20, 60), athlete_pts, 3)

        # Legend
        scout_leg = self.font_body.render("--- Scout Preference", True, (70, 130, 180))
        ath_leg = self.font_body.render("--- Athlete Profile", True, (220, 20, 60))
        self.screen.blit(scout_leg, (cx - 100, cy + radius + 40))
        self.screen.blit(ath_leg, (cx - 100, cy + radius + 65))

    def _draw_dashboard(self, info, offset_x, offset_y):
        # Action Card
        action = info["action_taken"]
        action_color = (50, 205, 50) if action == "Recommend" else (220, 20, 60)
        
        act_title = self.font_title.render("Agent Action:", True, self.text_color)
        act_text = self.font_large.render(action.upper(), True, action_color)
        self.screen.blit(act_title, (offset_x, offset_y))
        self.screen.blit(act_text, (offset_x, offset_y + 35))
        
        # Match Score
        score = info["match_score"]
        score_text = self.font_title.render(f"Calculated Match: {score * 100:.1f}%", True, self.text_color)
        self.screen.blit(score_text, (offset_x, offset_y + 120))
        
        # Reward
        reward = info["reward_earned"]
        rew_color = (50, 205, 50) if reward > 0 else (220, 20, 60)
        rew_text = self.font_title.render(f"Step Reward: {reward:+.1f}", True, rew_color)
        self.screen.blit(rew_text, (offset_x, offset_y + 180))
        
        # Step counter
        help_text = self.font_body.render("Close window or press Ctrl+C in terminal to exit", True, (150, 150, 150))
        self.screen.blit(help_text, (offset_x, self.height - 40))

    def close(self):
        pygame.quit()