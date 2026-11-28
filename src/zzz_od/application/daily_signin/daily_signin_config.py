import sys

from one_dragon.base.operation.application.application_config import ApplicationConfig


class DailySignInConfig(ApplicationConfig):

    def __init__(self, instance_idx: int, group_id: str) -> None:
        ApplicationConfig.__init__(self, 'daily_signin', instance_idx, group_id)

    @property
    def selected_sign(self) -> str:
        """未设置时，Mac 使用阿朔占卜。"""
        default = 'trigrams_collection' if sys.platform == 'darwin' else 'hou_hou_bakery'
        return self.get('selected_sign', default)

    @selected_sign.setter
    def selected_sign(self, value: str) -> None:
        self.update('selected_sign', value)
