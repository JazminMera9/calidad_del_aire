from django.urls import path

from . import views

app_name = "dashboard"

urlpatterns = [
    path("", views.inicio, name="inicio"),
    path("historial/", views.historial, name="historial"),
    path("explicabilidad/", views.explicabilidad, name="explicabilidad"),
    path("acerca/", views.acerca, name="acerca"),
]
