/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE617_Reachable_Assertion__connect_socket_66b.c
Label Definition File: CWE617_Reachable_Assertion.label.xml
Template File: sources-sink-66b.tmpl.c
*/
/*
 * @description
 * CWE: 617 Reachable Assertion
 * BadSource: connect_socket Read data using a connect socket (client side)
 * GoodSource: Number greater than ASSERT_VALUE
 * Sinks:
 *    BadSink : Assert if n is less than ASSERT_VALUE
 * Flow Variant: 66 Data flow: data passed in an array from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#include <assert.h>

#define ASSERT_VALUE 5

#ifdef _WIN32
# include <winsock2.h>
# include <windows.h>
# include <direct.h>
# pragma comment(lib, "ws2_32") /* include ws2_32.lib when linking */
# define CLOSE_SOCKET closesocket
#else /* NOT _WIN32 */
# define INVALID_SOCKET -1
# define SOCKET_ERROR -1
# define CLOSE_SOCKET close
# define SOCKET int
#endif

#define TCP_PORT 27015
#define CHAR_ARRAY_SIZE sizeof(data)*sizeof(data)

#ifndef OMITBAD

void CWE617_Reachable_Assertion__connect_socket_66b_bad_sink(int data_array[])
{
    /* copy data out of data_array */
    int data = data_array[2];
    /* POTENTIAL FLAW: this assertion could trigger if n < ASSERT_VALUE */
    assert(data > ASSERT_VALUE);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE617_Reachable_Assertion__connect_socket_66b_goodG2B_sink(int data_array[])
{
    int data = data_array[2];
    /* POTENTIAL FLAW: this assertion could trigger if n < ASSERT_VALUE */
    assert(data > ASSERT_VALUE);
}

#endif /* OMITGOOD */
