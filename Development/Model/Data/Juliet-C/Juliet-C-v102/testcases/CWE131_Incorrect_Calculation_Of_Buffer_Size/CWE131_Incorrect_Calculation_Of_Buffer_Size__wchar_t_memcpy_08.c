/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE131_Incorrect_Calculation_Of_Buffer_Size__wchar_t_memcpy_08.c
Label Definition File: CWE131_Incorrect_Calculation_Of_Buffer_Size.off_by_one.label.xml
Template File: sources-sink-08.tmpl.c
*/
/*
 * @description
 * CWE: 131 Incorrect Calculation of Buffer Size
 * BadSource:  Allocate memory for a string, but do not allocate space for NULL terminator
 * GoodSource: Allocate enough memory for a string and the NULL terminator
 * Sink: memcpy
 *    BadSink : Copy string to data using memcpy()
 * Flow Variant: 08 Control flow: if(static_returns_t()) and if(static_returns_f())
 *
 * */

#include "std_testcase.h"

/* MAINTENANCE NOTE: The length of this string should equal the 10 */
#define SRC_STRING L"AAAAAAAAAA"

/* The two function below always return the same value, so a tool
   should be able to identify that calls to the functions will always
   return a fixed value. */
static int static_returns_t()
{
    return 1;
}

static int static_returns_f()
{
    return 0;
}

#ifndef OMITBAD

void CWE131_Incorrect_Calculation_Of_Buffer_Size__wchar_t_memcpy_08_bad()
{
    wchar_t * data;
    data = NULL;
    if(static_returns_t())
    {
        /* FLAW: Did not leave space for a null terminator */
        data = (wchar_t *)malloc(10*sizeof(wchar_t));
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Allocate space for a null terminator */
        data = (wchar_t *)malloc((10+1)*sizeof(wchar_t));
    }
    {
        wchar_t data_src[10+1] = SRC_STRING;
        /* POTENTIAL FLAW: data may not have enough space to hold data_src */
        memcpy(data, data_src, wcslen(data_src)*sizeof(wchar_t));
        data[wcslen(data_src)] = L'\0'; /* NULL terminate */
        printWLine(data);
        free(data);
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B1() - use goodsource and badsink by changing the static_returns_t() to static_returns_f() */
static void goodG2B1()
{
    wchar_t * data;
    data = NULL;
    if(static_returns_f())
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FLAW: Did not leave space for a null terminator */
        data = (wchar_t *)malloc(10*sizeof(wchar_t));
    }
    else
    {
        /* FIX: Allocate space for a null terminator */
        data = (wchar_t *)malloc((10+1)*sizeof(wchar_t));
    }
    {
        wchar_t data_src[10+1] = SRC_STRING;
        /* POTENTIAL FLAW: data may not have enough space to hold data_src */
        memcpy(data, data_src, wcslen(data_src)*sizeof(wchar_t));
        data[wcslen(data_src)] = L'\0'; /* NULL terminate */
        printWLine(data);
        free(data);
    }
}

/* goodG2B2() - use goodsource and badsink by reversing the blocks in the if statement */
static void goodG2B2()
{
    wchar_t * data;
    data = NULL;
    if(static_returns_t())
    {
        /* FIX: Allocate space for a null terminator */
        data = (wchar_t *)malloc((10+1)*sizeof(wchar_t));
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FLAW: Did not leave space for a null terminator */
        data = (wchar_t *)malloc(10*sizeof(wchar_t));
    }
    {
        wchar_t data_src[10+1] = SRC_STRING;
        /* POTENTIAL FLAW: data may not have enough space to hold data_src */
        memcpy(data, data_src, wcslen(data_src)*sizeof(wchar_t));
        data[wcslen(data_src)] = L'\0'; /* NULL terminate */
        printWLine(data);
        free(data);
    }
}

void CWE131_Incorrect_Calculation_Of_Buffer_Size__wchar_t_memcpy_08_good()
{
    goodG2B1();
    goodG2B2();
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
    CWE131_Incorrect_Calculation_Of_Buffer_Size__wchar_t_memcpy_08_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE131_Incorrect_Calculation_Of_Buffer_Size__wchar_t_memcpy_08_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
