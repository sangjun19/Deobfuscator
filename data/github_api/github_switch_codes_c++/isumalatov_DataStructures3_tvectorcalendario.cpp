#include "tvectorcalendario.h"

int TVectorCalendario::daysInMonth(int m, int a)
{
    int days;
    switch (m)
    {
    case 2:
        if (a % 4 == 0 && (a % 100 != 0 || a % 400 == 0))
        {
            days = 29;
        }
        else
        {
            days = 28;
        }
        break;
    case 4:
    case 6:
    case 9:
    case 11:
        days = 30;
        break;
    default:
        days = 31;
        break;
    }
    return days;
}

void TVectorCalendario::copia(const TVectorCalendario &v)
{
    tamano = v.tamano;
    c = new TCalendario[tamano];
    for (int i = 0; i < tamano; i++)
    {
        c[i] = v.c[i];
    }
}

TVectorCalendario::TVectorCalendario()
{
    tamano = 0;
    c = NULL;
}

TVectorCalendario::TVectorCalendario(int tamano)
{
    if (tamano < 1)
    {
        tamano = 0;
        c = NULL;
    }
    else
    {
        tamano = tamano;
        c = new TCalendario[tamano];
    }
}

TVectorCalendario::TVectorCalendario(const TVectorCalendario &v)
{
    copia(v);
}

TVectorCalendario::~TVectorCalendario()
{
    if (c != NULL)
    {
        delete[] c;
        c = NULL;
    }
}

TVectorCalendario &TVectorCalendario::operator=(const TVectorCalendario &v)
{
    if (this != &v)
    {
        (*this).~TVectorCalendario();
        copia(v);
    }
    else
    {
        return *this;
    }

    return *this;
}

bool TVectorCalendario::operator==(const TVectorCalendario &v)
{
    if (tamano != v.tamano)
    {
        return false;
    }
    for (int i = 0; i < tamano; i++)
    {
        if (this->c[i] != v.c[i])
        {
            return false;
        }
    }
    return true;
}

bool TVectorCalendario::operator!=(const TVectorCalendario &v)
{
    return !(*this == v);
}

TCalendario &TVectorCalendario::operator[](int i)
{
    if (i > 0 && i <= this->tamano)
    {
        return this->c[i - 1];
    }

    return error;
}

TCalendario TVectorCalendario::operator[](int i) const
{
    if (i > 0 && i <= this->tamano)
    {
        return this->c[i - 1];
    }

    return error;
}

int TVectorCalendario::Tamano()
{
    return tamano;
}

int TVectorCalendario::Ocupadas()
{
    int ocupadas = 0;
    for (int i = 0; i < tamano; i++)
    {
        if (this->c[i] != this->error)
        {
            ocupadas++;
        }
    }
    return ocupadas;
}

bool TVectorCalendario::ExisteCal(const TCalendario &cal)
{
    for (int i = 0; i < tamano; i++)
    {
        if (c[i] == cal)
        {
            return true;
        }
    }
    return false;
}

void TVectorCalendario::MostrarMensajes(int d, int m, int a)
{
    if (m < 1 || m > 12 || a < 1900 || d < 1 || d > daysInMonth(m, a))
    {
        cout << "[]";
    }
    else
    {
        cout << "[";
        for (int i = 0; i < tamano; i++)
        {
            if (c[i].Dia() >= d && c[i].Mes() >= m && c[i].Anyo() >= a)
            {
                cout << c[i] << ", ";
            }
        }
        cout << "]";
    }
}

bool TVectorCalendario::Redimensionar(int t)
{
    if (t <= 0)
    {
        return false;
    }
    if (t == tamano)
    {
        return false;
    }
    if (t < tamano)
    {
        TCalendario *aux = new TCalendario[t];
        for (int i = 0; i < t; i++)
        {
            aux[i] = c[i];
        }
        delete[] c;
        c = aux;
        tamano = t;
        return true;
    }
    if (t > tamano)
    {
        TCalendario *aux = new TCalendario[t];
        for (int i = 0; i < tamano; i++)
        {
            aux[i] = c[i];
        }
        for (int i = tamano; i < t; i++)
        {
            aux[i] = TCalendario();
        }
        delete[] c;
        c = aux;
        tamano = t;
        return true;
    }
    return false;
}

ostream &operator<<(ostream &s, const TVectorCalendario &vec)
{

    s << "[";

    for (int i = 0; i < vec.tamano; i++)
    {

        s << "(" << i + 1 << ") " << vec.c[i];

        if (i < vec.tamano - 1)
        {
            s << ", ";
        }
    }

    s << "]";

    return s;
}