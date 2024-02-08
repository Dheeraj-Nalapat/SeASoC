/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE134_Uncontrolled_Format_String__char_connect_socket_fprintf_61a.c
Label Definition File: CWE134_Uncontrolled_Format_String.label.xml
Template File: sources-sinks-61a.tmpl.c
*/
/*
 * @description
 * CWE: 134 Uncontrolled Format String
 * BadSource: connect_socket Read data using a connect socket (client side)
 * GoodSource: Copy a fixed string into data
 * Sinks: fprintf
 *    GoodSink: fprintf with "%s" as the second argument and data as the third
 *    BadSink : fprintf with data as the second argument
 * Flow Variant: 61 Data flow: data returned from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#ifdef _WIN32
# include <winsock2.h>
# include <windows.h>
# include <direct.h>
# pragma comment(lib, "ws2_32") /* include ws2_32.lib when linking */
# define CLOSE_SOCKET closesocket
# define PATH_SZ 100
#else /* NOT _WIN32 */
# define INVALID_SOCKET -1
# define SOCKET_ERROR -1
# define CLOSE_SOCKET close
# define SOCKET int
# define PATH_SZ PATH_MAX
#endif

#define TCP_PORT 27015

#ifndef OMITBAD

/* bad function declaration */
char * CWE134_Uncontrolled_Format_String__char_connect_socket_fprintf_61b_bad_source(char * data);

void CWE134_Uncontrolled_Format_String__char_connect_socket_fprintf_61_bad()
{
    char * data;
    char data_buf[100] = "";
    data = data_buf;
    data = CWE134_Uncontrolled_Format_String__char_connect_socket_fprintf_61b_bad_source(data);
    /* POTENTIAL FLAW: Do not specify the format allowing a possible format string vulnerability */
    fprintf(stdout, data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
char * CWE134_Uncontrolled_Format_String__char_connect_socket_fprintf_61b_goodG2B_source(char * data);

static void goodG2B()
{
    char * data;
    char data_buf[100] = "";
    data = data_buf;
    data = CWE134_Uncontrolled_Format_String__char_connect_socket_fprintf_61b_goodG2B_source(data);
    /* POTENTIAL FLAW: Do not specify the format allowing a possible format string vulnerability */
    fprintf(stdout, data);
}

/* goodB2G uses the BadSource with the GoodSink */
char * CWE134_Uncontrolled_Format_String__char_connect_socket_fprintf_61b_goodB2G_source(char * data);

static void goodB2G()
{
    char * data;
    char data_buf[100] = "";
    data = data_buf;
    data = CWE134_Uncontrolled_Format_String__char_connect_socket_fprintf_61b_goodB2G_source(data);
    /* FIX: Specify the format disallowing a format string vulnerability */
    fprintf(stdout, "%s\n", data);
}

void CWE134_Uncontrolled_Format_String__char_connect_socket_fprintf_61_good()
{
    goodG2B();
    goodB2G();
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
    CWE134_Uncontrolled_Format_String__char_connect_socket_fprintf_61_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE134_Uncontrolled_Format_String__char_connect_socket_fprintf_61_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif