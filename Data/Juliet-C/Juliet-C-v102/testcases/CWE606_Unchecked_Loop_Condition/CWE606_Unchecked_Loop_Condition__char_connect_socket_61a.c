/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE606_Unchecked_Loop_Condition__char_connect_socket_61a.c
Label Definition File: CWE606_Unchecked_Loop_Condition.label.xml
Template File: sources-sinks-61a.tmpl.c
*/
/*
 * @description
 * CWE: 606 Unchecked Input For Loop Condition
 * BadSource: connect_socket Read data using a connect socket (client side)
 * GoodSource: Input a number less than MAX_LOOP
 * Sinks:
 *    GoodSink: Use data as the for loop variant after checking to see if it is less than MAX_LOOP
 *    BadSink : Use data as the for loop variant without checking its size
 * Flow Variant: 61 Data flow: data returned from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#define MAX_LOOP 10000

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
char * CWE606_Unchecked_Loop_Condition__char_connect_socket_61b_bad_source(char * data);

void CWE606_Unchecked_Loop_Condition__char_connect_socket_61_bad()
{
    char * data;
    char data_buf[100] = "";
    data = data_buf;
    data = CWE606_Unchecked_Loop_Condition__char_connect_socket_61b_bad_source(data);
    {
        int i, n, v;
        if (sscanf(data, "%d", &n) == 1)
        {
            /* POTENTIAL FLAW: user-supplied value 'n' could lead to very large loop iteration */
            v = 0;
            for (i = 0; i < n; i++)
            {
                /* INCIDENTAL: CWE 561: Dead Code - non-avoidable if n <= 0 */
                v++; /* avoid a dead/empty code block issue */
            }
            printIntLine(v);
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
char * CWE606_Unchecked_Loop_Condition__char_connect_socket_61b_goodG2B_source(char * data);

static void goodG2B()
{
    char * data;
    char data_buf[100] = "";
    data = data_buf;
    data = CWE606_Unchecked_Loop_Condition__char_connect_socket_61b_goodG2B_source(data);
    {
        int i, n, v;
        if (sscanf(data, "%d", &n) == 1)
        {
            /* POTENTIAL FLAW: user-supplied value 'n' could lead to very large loop iteration */
            v = 0;
            for (i = 0; i < n; i++)
            {
                /* INCIDENTAL: CWE 561: Dead Code - non-avoidable if n <= 0 */
                v++; /* avoid a dead/empty code block issue */
            }
            printIntLine(v);
        }
    }
}

/* goodB2G uses the BadSource with the GoodSink */
char * CWE606_Unchecked_Loop_Condition__char_connect_socket_61b_goodB2G_source(char * data);

static void goodB2G()
{
    char * data;
    char data_buf[100] = "";
    data = data_buf;
    data = CWE606_Unchecked_Loop_Condition__char_connect_socket_61b_goodB2G_source(data);
    {
        int i, n, v;
        if (sscanf(data, "%d", &n) == 1)
        {
            /* FIX: limit loop iteration counts */
            if (n < MAX_LOOP)
            {
                v = 0;
                for (i = 0; i < n; i++)
                {
                    /* INCIDENTAL: CWE 561: Dead Code - non-avoidable if n <= 0 */
                    v++; /* avoid a dead/empty code block issue */
                }
                printIntLine(v);
            }
        }
    }
}

void CWE606_Unchecked_Loop_Condition__char_connect_socket_61_good()
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
    CWE606_Unchecked_Loop_Condition__char_connect_socket_61_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE606_Unchecked_Loop_Condition__char_connect_socket_61_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
