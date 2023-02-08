import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches

class Drawer:
    def __init__(self, node_width=5, node_height=1, vertical_gap=1, canvas_width=12, fontsize=10):
        self.node_width = node_width
        self.node_height = node_height
        self.vertical_gap = vertical_gap
        self.canvas_width = canvas_width
        self.fontsize = fontsize

    def draw(self, ax, tree):
        bbox = self.grow_node(ax, tree, -1, 0, 0, self.canvas_width)
        bbox[1] = bbox[1] + 0.1 * self.node_width
        bbox[2] = bbox[2] - 0.1 * self.node_height
        ax.axis('off')
        return bbox

    def draw_internal_node(self, ax, tree, i, pos_x, pos_y, w):
        d = tree.feature[i]
        v = tree.threshold[i]
        ax.annotate('if X[%d] <= %.3f' % (d, v), (pos_x + 0.5 * self.node_width, pos_y + 0.5 * self.node_height), fontsize=self.fontsize, ha='center', va='center', zorder=200)

    def draw_leaf_node(self, ax, tree, i, pos_x, pos_y, w):
        y = np.argmax(tree.value[i])
        py = tree.value[i] / sum(tree.value[i])
        t = 'Y = %d\n' % (y,)
        t = t + 'P[Y|X] = [%.2f, ' % (py[0],) + ', '.join(['%.2f' % (v,) for v in py[1:]]) + ']'
        t = t + '\nN[Y|X] = [%d, ' % (tree.value[i][0],)  + ', '.join(['%d' % (v,) for v in tree.value[i][1:]]) + ']'
        ax.annotate(t, (pos_x + 0.5 * self.node_width, pos_y + 0.5 * self.node_height), fontsize=self.fontsize, ha='center', va='center', zorder=200)

    def draw_edge(self, ax, tree, i, pos_x, pos_y, w, side='l'):
        [s, t] = [-1, 'Yes'] if side == 'l' else [1, 'No']
        source_x = pos_x + 0.5 * self.node_width
        width_x = 0.5 * s * w
        gap_y = self.vertical_gap
        ax.plot([source_x, source_x + width_x], [pos_y, pos_y - gap_y], 'k-')
        ax.add_patch(
                patches.Ellipse(
                    (source_x + 0.5 * width_x, pos_y - 0.5 * gap_y),
                    0.5*self.node_width,
                    0.5*self.node_height,
                    edgecolor = 'k',
                    facecolor = 'w',
                    fill = True,
                    zorder=100
            ))
        ax.annotate(t, (source_x + 0.5 * width_x, pos_y - 0.5 * gap_y), fontsize=self.fontsize, ha='center', va='center', zorder=200)

    def grow_node(self, ax, tree, i, pos_x, pos_y, w):
        ax.add_patch(
            patches.Rectangle(
                (pos_x, pos_y),
                self.node_width,
                self.node_height,
                edgecolor = 'k',
                facecolor = 'w',
                fill = True
        ))
        d = tree.feature[i]
        v = tree.threshold[i]
        flag = True
        if tree.children_left[i] >= 0:
            pos_x_l, pos_x_r = pos_x - 0.5 * w, pos_x + 0.5 * w
            pos_y_next = pos_y - self.node_height - self.vertical_gap
            w_next = 0.5 * w
            bbox_left = self.grow_node(ax, tree, tree.children_left[i], pos_x_l, pos_y_next, w_next)
            bbox_right = self.grow_node(ax, tree, tree.children_right[i], pos_x_r, pos_y_next, w_next)
            self.draw_edge(ax, tree, i, pos_x, pos_y, w, side='l')
            self.draw_edge(ax, tree, i, pos_x, pos_y, w, side='r')
            flag = False
        if flag:
            self.draw_leaf_node(ax, tree, i, pos_x, pos_y, w)
            bbox = [pos_x, pos_x + self.node_width, pos_y, pos_y + self.node_height]
        else:
            self.draw_internal_node(ax, tree, i, pos_x, pos_y, w)
            bbox = [bbox_left[0], bbox_right[1], min(bbox_left[2], bbox_right[2]), pos_y + self.node_height]
        return bbox