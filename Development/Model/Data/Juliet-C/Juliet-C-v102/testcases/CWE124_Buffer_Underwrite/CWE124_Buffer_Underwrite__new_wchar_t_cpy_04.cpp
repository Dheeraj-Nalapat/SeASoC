/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE124_Buffer_Underwrite__new_wchar_t_cpy_04.cpp
Label Definition File: CWE124_Buffer_Underwrite__new.label.xml
Template File: sources-sink-04.tmpl.cpp
*/
/*
 * @description
 * CWE: 124 Buffer Underwrite
 * BadSource:  Set data pointer to before the allocated memory buffer
 * GoodSource: Set data pointer to the allocated memory buffer
 * Sink: cpy
 *    BadSink : Copy string to data using wcscpy
 * Flow Variant: 04 Control flow: if(static_const_t) and if(static_const_f)
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

/* The two variables below are declared "const", so a tool should
   be able to identify that reads of these will always return their
   initialized values. */
static const int static_const_t = 1; /* true */
static const int static_const_f = 0; /* false */

namespace CWE124_Buffer_Underwrite__new_wchar_t_cpy_04
{

#ifndef OMITBAD

void bad()
{
    wchar_t * data;
    data = NULL;
    if(static_const_t)
    {
        {
            wchar_t * data_buf = new wchar_t[100];
            wmemset(data_buf, L'A', 100-1);
            data_buf[100-1] = L'\0';
            /* FLAW: Set data pointer to before the allocated memory buffer */
            data = data_buf - 8;
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            wchar_t * data_buf = new wchar_t[100];
            wmemset(data_buf, L'A', 100-1);
            data_buf[100-1] = L'\0';
            /* FIX: Set data pointer to the allocated memory buffer */
            data = data_buf;
        }
    }
    {
        wchar_t src[100];
        wmemset(src, L'C', 100-1); /* fill with 'C's */
        src[100-1] = L'\0'; /* null terminate */
        /* POTENTIAL FLAW: Possibly copying data to memory before the destination buffer */
        wcscpy(data, src);
        printWLine(data);
        /* INCIDENTAL CWE-401: Memory Leak - data may not point to location
           returned by new [] so can't safely call delete [] on it */
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B1() - use goodsource and badsink by changing the static_const_t to static_const_f */
static void goodG2B1()
{
    wchar_t * data;
    data = NULL;
    if(static_const_f)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            wchar_t * data_buf = new wchar_t[100];
            wmemset(data_buf, L'A', 100-1);
            data_buf[100-1] = L'\0';
            /* FLAW: Set data pointer to before the allocated memory buffer */
            data = data_buf - 8;
        }
    }
    else
    {
        {
            wchar_t * data_buf = new wchar_t[100];
            wmemset(data_buf, L'A', 100-1);
            data_buf[100-1] = L'\0';
            /* FIX: Set data pointer to the allocated memory buffer */
            data = data_buf;
        }
    }
    {
        wchar_t src[100];
        wmemset(src, L'C', 100-1); /* fill with 'C's */
        src[100-1] = L'\0'; /* null terminate */
        /* POTENTIAL FLAW: Possibly copying data to memory before the destination buffer */
        wcscpy(data, src);
        printWLine(data);
        /* INCIDENTAL CWE-401: Memory Leak - data may not point to location
           returned by new [] so can't safely call delete [] on it */
    }
}

/* goodG2B2() - use goodsource and badsink by reversing the blocks in the if statement */
static void goodG2B2()
{
    wchar_t * data;
    data = NULL;
    if(static_const_t)
    {
        {
            wchar_t * data_buf = new wchar_t[100];
            wmemset(data_buf, L'A', 100-1);
            data_buf[100-1] = L'\0';
            /* FIX: Set data pointer to the allocated memory buffer */
            data = data_buf;
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            wchar_t * data_buf = new wchar_t[100];
            wmemset(data_buf, L'A', 100-1);
            data_buf[100-1] = L'\0';
            /* FLAW: Set data pointer to before the allocated memory buffer */
            data = data_buf - 8;
        }
    }
    {
        wchar_t src[100];
        wmemset(src, L'C', 100-1); /* fill with 'C's */
        src[100-1] = L'\0'; /* null terminate */
        /* POTENTIAL FLAW: Possibly copying data to memory before the destination buffer */
        wcscpy(data, src);
        printWLine(data);
        /* INCIDENTAL CWE-401: Memory Leak - data may not point to location
           returned by new [] so can't safely call delete [] on it */
    }
}

void good()
{
    goodG2B1();
    goodG2B2();
}

#endif /* OMITGOOD */

} // close namespace

/* Below is the main(). It is only used when building this testcase on
   its own for testing or for building a binary to use in testing binary
   analysis tools. It is not used when compiling all the testcases as one
   application, which is how source code analysis tools are tested. */

#ifdef INCLUDEMAIN

using namespace CWE124_Buffer_Underwrite__new_wchar_t_cpy_04; // so that we can use good and bad easily

int main(int argc, char * argv[])
{
    /* seed randomness */
    srand( (unsigned)time(NULL) );
#ifndef OMITGOOD
    printLine("Calling good()...");
    good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif