from django.urls import path
from .views import predict_image, signup, login,all_predictions,logout

urlpatterns = [
    path("predict/", predict_image,name="predict_image"),
    path("signup/", signup),
    path("login/", login),
    path("logout/", logout),
    path("all/", all_predictions, name="all_predictions"),
]
