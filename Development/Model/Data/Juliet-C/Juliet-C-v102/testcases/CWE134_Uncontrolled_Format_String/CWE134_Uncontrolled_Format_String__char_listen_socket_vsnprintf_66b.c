/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE134_Uncontrolled_Format_String__char_listen_socket_vsnprintf_66b.c
Label Definition File: CWE134_Uncontrolled_Format_String.vasinks.label.xml
Template File: sources-vasinks-66b.tmpl.c
*/
/*
 * @description
 * CWE: 134 Uncontrolled Format String
 * BadSource: listen_socket Read data using a listen socket (server side)
 * GoodSource: Copy a fixed string into data
 * Sinks: vsnprintf
 *    GoodSink: vsnprintf with a format string
 *    BadSink : vsnprintf without a format string
 * Flow Variant: 66 Data flow: data passed in an array from one function to another in different source files
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

#ifndef OMITBAD

static void bad_vasink(char * data, ...)
{
    {
        char dest[100] = "";
        va_list args;
        va_start(args, data);
        /* POTENTIAL FLAW: Do not specify the format allowing a possible format string vulnerability */
        vsnprintf(dest, 100-1, data, args);
        va_end(args);
        printLine(dest);
    }
}

void CWE134_Uncontrolled_Format_String__char_listen_socket_vsnprintf_66b_bad_sink(char * data_array[])
{
    /* copy data out of data_array */
    char * data = data_array[2];
    bad_vasink(data, data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
static void goodG2B_vasink(char * data, ...)
{
    {
        char dest[100] = "";
        va_list args;
        va_start(args, data);
        /* POTENTIAL FLAW: Do not specify the format allowing a possible format string vulnerability */
        vsnprintf(dest, 100-1, data, args);
        va_end(args);
        printLine(dest);
    }
}

void CWE134_Uncontrolled_Format_String__char_listen_socket_vsnprintf_66b_goodG2B_sink(char * data_array[])
{
    char * data = data_array[2];
    goodG2B_vasink(data, data);
}

/* goodB2G uses the BadSource with the GoodSink */
static void goodB2G_vasink(char * data, ...)
{
    {
        char dest[100] = "";
        va_list args;
        va_start(args, data);
        /* FIX: Specify the format disallowing a format string vulnerability */
        vsnprintf(dest, 100-1, "%s", args);
        va_end(args);
        printLine(dest);
    }
}

void CWE134_Uncontrolled_Format_String__char_listen_socket_vsnprintf_66b_goodB2G_sink(char * data_array[])
{
    char * data = data_array[2];
    goodB2G_vasink(data, data);
}

#endif /* OMITGOOD */
