/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE134_Uncontrolled_Format_String__char_listen_socket_vfprintf_04.c
Label Definition File: CWE134_Uncontrolled_Format_String.vasinks.label.xml
Template File: sources-vasinks-04.tmpl.c
*/
/*
 * @description
 * CWE: 134 Uncontrolled Format String
 * BadSource: listen_socket Read data using a listen socket (server side)
 * GoodSource: Copy a fixed string into data
 * Sinks: vfprintf
 *    GoodSink: vfprintf with a format string
 *    BadSink : vfprintf without a format string
 * Flow Variant: 04 Control flow: if(static_const_t) and if(static_const_f)
 *
 * */

#include <stdarg.h>
#include "std_testcase.h"

#ifdef _WIN32
# include <winsock2.h>
# include <windows.h>
# include <direct.h>
# define PATH_SZ 100
# pragma comment(lib, "ws2_32") /* include ws2_32.lib when linking */
# define CLOSE_SOCKET closesocket
#else
# define PATH_SZ PATH_MAX
# define INVALID_SOCKET -1
# define SOCKET_ERROR -1
# define CLOSE_SOCKET close
# define SOCKET int
#endif

#define TCP_PORT 27015
#define LISTEN_BACKLOG 5

/* The two variables below are declared "const", so a tool should
   be able to identify that reads of these will always return their
   initialized values. */
static const int static_const_t = 1; /* true */
static const int static_const_f = 0; /* false */

#ifndef OMITBAD

static void bad_vasinkb(char * data, ...)
{
    {
        va_list args;
        va_start(args, data);
        /* POTENTIAL FLAW: Do not specify the format allowing a possible format string vulnerability */
        vfprintf(stdout, data, args);
        va_end(args);
    }
}

/* INCIDENTAL: CWE 561 Dead Code, the function below is never called */
static void bad_vasinkg(char * data, ...)
{
    {
        va_list args;
        va_start(args, data);
        /* FIX: Specify the format disallowing a format string vulnerability */
        vfprintf(stdout, "%s", args);
        va_end(args);
    }
}

void CWE134_Uncontrolled_Format_String__char_listen_socket_vfprintf_04_bad()
{
    char * data;
    char data_buf[100] = "";
    data = data_buf;
    if(static_const_t)
    {
        {
#ifdef _WIN32
            WSADATA wsa_data;
            int wsa_data_init = 0;
#endif
            int recv_rv;
            struct sockaddr_in s_in;
            char *replace;
            SOCKET listen_socket = INVALID_SOCKET;
            SOCKET accept_socket = INVALID_SOCKET;
            size_t data_len = strlen(data);
            do
            {
#ifdef _WIN32
                if (WSAStartup(MAKEWORD(2,2), &wsa_data) != NO_ERROR) break;
                wsa_data_init = 1;
#endif
                listen_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
                if (listen_socket == INVALID_SOCKET) break;
                memset(&s_in, 0, sizeof(s_in));
                s_in.sin_family = AF_INET;
                s_in.sin_addr.s_addr = INADDR_ANY;
                s_in.sin_port = htons(TCP_PORT);
                if (bind(listen_socket, (struct sockaddr*)&s_in, sizeof(s_in)) == SOCKET_ERROR) break;
                if (listen(listen_socket, LISTEN_BACKLOG) == SOCKET_ERROR) break;
                accept_socket = accept(listen_socket, NULL, NULL);
                if (accept_socket == SOCKET_ERROR) break;
                /* Abort on error or the connection was closed */
                recv_rv = recv(accept_socket, (char *)data+data_len, (int)(100-data_len-1), 0);
                if (recv_rv == SOCKET_ERROR || recv_rv == 0) break;
                /* Append null terminator */
                data[recv_rv] = '\0';
                /* Eliminate CRLF */
                replace = strchr(data, '\r');
                if (replace) *replace = '\0';
                replace = strchr(data, '\n');
                if (replace) *replace = '\0';
            }
            while (0);
            if (listen_socket != INVALID_SOCKET) CLOSE_SOCKET(listen_socket);
            if (accept_socket != INVALID_SOCKET) CLOSE_SOCKET(accept_socket);
#ifdef _WIN32
            if (wsa_data_init) WSACleanup();
#endif
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Use a fixed string that does not contain a format specifier */
        strcpy(data, "fixedstringtest");
    }
    if(static_const_t)
    {
        bad_vasinkb(data, data);
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        bad_vasinkg(data, data);
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* INCIDENTAL: CWE 561 Dead Code, the function below is never called */
static void goodB2G1_vasinkb(char * data, ...)
{
    {
        va_list args;
        va_start(args, data);
        /* POTENTIAL FLAW: Do not specify the format allowing a possible format string vulnerability */
        vfprintf(stdout, data, args);
        va_end(args);
    }
}

static void goodB2G1_vasinkg(char * data, ...)
{
    {
        va_list args;
        va_start(args, data);
        /* FIX: Specify the format disallowing a format string vulnerability */
        vfprintf(stdout, "%s", args);
        va_end(args);
    }
}

/* goodB2G1() - use badsource and goodsink by changing the second static_const_t to static_const_f */
static void goodB2G1()
{
    char * data;
    char data_buf[100] = "";
    data = data_buf;
    if(static_const_t)
    {
        {
#ifdef _WIN32
            WSADATA wsa_data;
            int wsa_data_init = 0;
#endif
            int recv_rv;
            struct sockaddr_in s_in;
            char *replace;
            SOCKET listen_socket = INVALID_SOCKET;
            SOCKET accept_socket = INVALID_SOCKET;
            size_t data_len = strlen(data);
            do
            {
#ifdef _WIN32
                if (WSAStartup(MAKEWORD(2,2), &wsa_data) != NO_ERROR) break;
                wsa_data_init = 1;
#endif
                listen_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
                if (listen_socket == INVALID_SOCKET) break;
                memset(&s_in, 0, sizeof(s_in));
                s_in.sin_family = AF_INET;
                s_in.sin_addr.s_addr = INADDR_ANY;
                s_in.sin_port = htons(TCP_PORT);
                if (bind(listen_socket, (struct sockaddr*)&s_in, sizeof(s_in)) == SOCKET_ERROR) break;
                if (listen(listen_socket, LISTEN_BACKLOG) == SOCKET_ERROR) break;
                accept_socket = accept(listen_socket, NULL, NULL);
                if (accept_socket == SOCKET_ERROR) break;
                /* Abort on error or the connection was closed */
                recv_rv = recv(accept_socket, (char *)data+data_len, (int)(100-data_len-1), 0);
                if (recv_rv == SOCKET_ERROR || recv_rv == 0) break;
                /* Append null terminator */
                data[recv_rv] = '\0';
                /* Eliminate CRLF */
                replace = strchr(data, '\r');
                if (replace) *replace = '\0';
                replace = strchr(data, '\n');
                if (replace) *replace = '\0';
            }
            while (0);
            if (listen_socket != INVALID_SOCKET) CLOSE_SOCKET(listen_socket);
            if (accept_socket != INVALID_SOCKET) CLOSE_SOCKET(accept_socket);
#ifdef _WIN32
            if (wsa_data_init) WSACleanup();
#endif
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Use a fixed string that does not contain a format specifier */
        strcpy(data, "fixedstringtest");
    }
    if(static_const_f)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        goodB2G1_vasinkb(data, data);
    }
    else
    {
        goodB2G1_vasinkg(data, data);
    }
}

/* INCIDENTAL: CWE 561 Dead Code, the function below is never called */
static void goodB2G2_vasinkb(char * data, ...)
{
    {
        va_list args;
        va_start(args, data);
        /* POTENTIAL FLAW: Do not specify the format allowing a possible format string vulnerability */
        vfprintf(stdout, data, args);
        va_end(args);
    }
}

static void goodB2G2_vasinkg(char * data, ...)
{
    {
        va_list args;
        va_start(args, data);
        /* FIX: Specify the format disallowing a format string vulnerability */
        vfprintf(stdout, "%s", args);
        va_end(args);
    }
}

/* goodB2G2() - use badsource and goodsink by reversing the blocks in the second if */
static void goodB2G2()
{
    char * data;
    char data_buf[100] = "";
    data = data_buf;
    if(static_const_t)
    {
        {
#ifdef _WIN32
            WSADATA wsa_data;
            int wsa_data_init = 0;
#endif
            int recv_rv;
            struct sockaddr_in s_in;
            char *replace;
            SOCKET listen_socket = INVALID_SOCKET;
            SOCKET accept_socket = INVALID_SOCKET;
            size_t data_len = strlen(data);
            do
            {
#ifdef _WIN32
                if (WSAStartup(MAKEWORD(2,2), &wsa_data) != NO_ERROR) break;
                wsa_data_init = 1;
#endif
                listen_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
                if (listen_socket == INVALID_SOCKET) break;
                memset(&s_in, 0, sizeof(s_in));
                s_in.sin_family = AF_INET;
                s_in.sin_addr.s_addr = INADDR_ANY;
                s_in.sin_port = htons(TCP_PORT);
                if (bind(listen_socket, (struct sockaddr*)&s_in, sizeof(s_in)) == SOCKET_ERROR) break;
                if (listen(listen_socket, LISTEN_BACKLOG) == SOCKET_ERROR) break;
                accept_socket = accept(listen_socket, NULL, NULL);
                if (accept_socket == SOCKET_ERROR) break;
                /* Abort on error or the connection was closed */
                recv_rv = recv(accept_socket, (char *)data+data_len, (int)(100-data_len-1), 0);
                if (recv_rv == SOCKET_ERROR || recv_rv == 0) break;
                /* Append null terminator */
                data[recv_rv] = '\0';
                /* Eliminate CRLF */
                replace = strchr(data, '\r');
                if (replace) *replace = '\0';
                replace = strchr(data, '\n');
                if (replace) *replace = '\0';
            }
            while (0);
            if (listen_socket != INVALID_SOCKET) CLOSE_SOCKET(listen_socket);
            if (accept_socket != INVALID_SOCKET) CLOSE_SOCKET(accept_socket);
#ifdef _WIN32
            if (wsa_data_init) WSACleanup();
#endif
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Use a fixed string that does not contain a format specifier */
        strcpy(data, "fixedstringtest");
    }
    if(static_const_t)
    {
        goodB2G2_vasinkg(data, data);
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        goodB2G2_vasinkb(data, data);
    }
}

/* INCIDENTAL: CWE 561 Dead Code, the function below is never called */
static void goodG2B1_vasinkg(char * data, ...)
{
    {
        va_list args;
        va_start(args, data);
        /* FIX: Specify the format disallowing a format string vulnerability */
        vfprintf(stdout, "%s", args);
        va_end(args);
    }
}

static void goodG2B1_vasinkb(char * data, ...)
{
    {
        va_list args;
        va_start(args, data);
        /* POTENTIAL FLAW: Do not specify the format allowing a possible format string vulnerability */
        vfprintf(stdout, data, args);
        va_end(args);
    }
}

/* goodG2B1() - use goodsource and badsink by changing the first static_const_t to static_const_f */
static void goodG2B1()
{
    char * data;
    char data_buf[100] = "";
    data = data_buf;
    if(static_const_f)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
#ifdef _WIN32
            WSADATA wsa_data;
            int wsa_data_init = 0;
#endif
            int recv_rv;
            struct sockaddr_in s_in;
            char *replace;
            SOCKET listen_socket = INVALID_SOCKET;
            SOCKET accept_socket = INVALID_SOCKET;
            size_t data_len = strlen(data);
            do
            {
#ifdef _WIN32
                if (WSAStartup(MAKEWORD(2,2), &wsa_data) != NO_ERROR) break;
                wsa_data_init = 1;
#endif
                listen_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
                if (listen_socket == INVALID_SOCKET) break;
                memset(&s_in, 0, sizeof(s_in));
                s_in.sin_family = AF_INET;
                s_in.sin_addr.s_addr = INADDR_ANY;
                s_in.sin_port = htons(TCP_PORT);
                if (bind(listen_socket, (struct sockaddr*)&s_in, sizeof(s_in)) == SOCKET_ERROR) break;
                if (listen(listen_socket, LISTEN_BACKLOG) == SOCKET_ERROR) break;
                accept_socket = accept(listen_socket, NULL, NULL);
                if (accept_socket == SOCKET_ERROR) break;
                /* Abort on error or the connection was closed */
                recv_rv = recv(accept_socket, (char *)data+data_len, (int)(100-data_len-1), 0);
                if (recv_rv == SOCKET_ERROR || recv_rv == 0) break;
                /* Append null terminator */
                data[recv_rv] = '\0';
                /* Eliminate CRLF */
                replace = strchr(data, '\r');
                if (replace) *replace = '\0';
                replace = strchr(data, '\n');
                if (replace) *replace = '\0';
            }
            while (0);
            if (listen_socket != INVALID_SOCKET) CLOSE_SOCKET(listen_socket);
            if (accept_socket != INVALID_SOCKET) CLOSE_SOCKET(accept_socket);
#ifdef _WIN32
            if (wsa_data_init) WSACleanup();
#endif
        }
    }
    else
    {
        /* FIX: Use a fixed string that does not contain a format specifier */
        strcpy(data, "fixedstringtest");
    }
    if(static_const_t)
    {
        goodG2B1_vasinkb(data, data);
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        goodG2B1_vasinkg(data, data);
    }
}

/* INCIDENTAL: CWE 561 Dead Code, the function below is never called */
static void goodG2B2_vasinkg(char * data, ...)
{
    {
        va_list args;
        va_start(args, data);
        /* FIX: Specify the format disallowing a format string vulnerability */
        vfprintf(stdout, "%s", args);
        va_end(args);
    }
}

static void goodG2B2_vasinkb(char * data, ...)
{
    {
        va_list args;
        va_start(args, data);
        /* POTENTIAL FLAW: Do not specify the format allowing a possible format string vulnerability */
        vfprintf(stdout, data, args);
        va_end(args);
    }
}

/* goodG2B2() - use goodsource and badsink by reversing the blocks in the first if */
static void goodG2B2()
{
    char * data;
    char data_buf[100] = "";
    data = data_buf;
    if(static_const_t)
    {
        /* FIX: Use a fixed string that does not contain a format specifier */
        strcpy(data, "fixedstringtest");
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
#ifdef _WIN32
            WSADATA wsa_data;
            int wsa_data_init = 0;
#endif
            int recv_rv;
            struct sockaddr_in s_in;
            char *replace;
            SOCKET listen_socket = INVALID_SOCKET;
            SOCKET accept_socket = INVALID_SOCKET;
            size_t data_len = strlen(data);
            do
            {
#ifdef _WIN32
                if (WSAStartup(MAKEWORD(2,2), &wsa_data) != NO_ERROR) break;
                wsa_data_init = 1;
#endif
                listen_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
                if (listen_socket == INVALID_SOCKET) break;
                memset(&s_in, 0, sizeof(s_in));
                s_in.sin_family = AF_INET;
                s_in.sin_addr.s_addr = INADDR_ANY;
                s_in.sin_port = htons(TCP_PORT);
                if (bind(listen_socket, (struct sockaddr*)&s_in, sizeof(s_in)) == SOCKET_ERROR) break;
                if (listen(listen_socket, LISTEN_BACKLOG) == SOCKET_ERROR) break;
                accept_socket = accept(listen_socket, NULL, NULL);
                if (accept_socket == SOCKET_ERROR) break;
                /* Abort on error or the connection was closed */
                recv_rv = recv(accept_socket, (char *)data+data_len, (int)(100-data_len-1), 0);
                if (recv_rv == SOCKET_ERROR || recv_rv == 0) break;
                /* Append null terminator */
                data[recv_rv] = '\0';
                /* Eliminate CRLF */
                replace = strchr(data, '\r');
                if (replace) *replace = '\0';
                replace = strchr(data, '\n');
                if (replace) *replace = '\0';
            }
            while (0);
            if (listen_socket != INVALID_SOCKET) CLOSE_SOCKET(listen_socket);
            if (accept_socket != INVALID_SOCKET) CLOSE_SOCKET(accept_socket);
#ifdef _WIN32
            if (wsa_data_init) WSACleanup();
#endif
        }
    }
    if(static_const_t)
    {
        goodG2B2_vasinkb(data, data);
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        goodG2B2_vasinkg(data, data);
    }
}

void CWE134_Uncontrolled_Format_String__char_listen_socket_vfprintf_04_good()
{
    goodG2B1();
    goodG2B2();
    goodB2G1();
    goodB2G2();
}

#endif /* OMITGOOD */

/* Below is the main(). It is only used when building this testcase on
   its own for testing or for building a binary to use in testing binary
   analysis tools. It is not used when compiling all the testcases as one
   application, which is how source code analysis tools are tested. */

#ifdef INCLUDEMAIN

int main(int argc, char * argv[])
{
    /* seed randomness */
    srand( (unsigned)time(NULL) );
#ifndef OMITGOOD
    printLine("Calling good()...");
    CWE134_Uncontrolled_Format_String__char_listen_socket_vfprintf_04_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE134_Uncontrolled_Format_String__char_listen_socket_vfprintf_04_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
