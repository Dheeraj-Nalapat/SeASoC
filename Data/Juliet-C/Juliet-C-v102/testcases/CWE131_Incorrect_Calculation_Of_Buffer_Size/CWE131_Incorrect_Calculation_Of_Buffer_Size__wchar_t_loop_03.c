/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE131_Incorrect_Calculation_Of_Buffer_Size__wchar_t_loop_03.c
Label Definition File: CWE131_Incorrect_Calculation_Of_Buffer_Size.off_by_one.label.xml
Template File: sources-sink-03.tmpl.c
*/
/*
 * @description
 * CWE: 131 Incorrect Calculation of Buffer Size
 * BadSource:  Allocate memory for a string, but do not allocate space for NULL terminator
 * GoodSource: Allocate enough memory for a string and the NULL terminator
 * Sink: loop
 *    BadSink : Copy array to data using a loop
 * Flow Variant: 03 Control flow: if(5==5) and if(5!=5)
 *
 * */

#include "std_testcase.h"

/* MAINTENANCE NOTE: The length of this string should equal the 10 */
#define SRC_STRING L"AAAAAAAAAA"

#ifndef OMITBAD

void CWE131_Incorrect_Calculation_Of_Buffer_Size__wchar_t_loop_03_bad()
{
    wchar_t * data;
    data = NULL;
    if(5==5)
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
        size_t i, src_len;
        src_len = wcslen(data_src);
        /* POTENTIAL FLAW: data may not have enough space to hold data_src */
        for (i = 0; i < src_len; i++)
        {
            data[i] = data_src[i];
        }
        data[wcslen(data_src)] = L'\0'; /* NULL terminate */
        printWLine(data);
        free(data);
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B1() - use goodsource and badsink by changing the 5==5 to 5!=5 */
static void goodG2B1()
{
    wchar_t * data;
    data = NULL;
    if(5!=5)
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
        size_t i, src_len;
        src_len = wcslen(data_src);
        /* POTENTIAL FLAW: data may not have enough space to hold data_src */
        for (i = 0; i < src_len; i++)
        {
            data[i] = data_src[i];
        }
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
    if(5==5)
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
        size_t i, src_len;
        src_len = wcslen(data_src);
        /* POTENTIAL FLAW: data may not have enough space to hold data_src */
        for (i = 0; i < src_len; i++)
        {
            data[i] = data_src[i];
        }
        data[wcslen(data_src)] = L'\0'; /* NULL terminate */
        printWLine(data);
        free(data);
    }
}

void CWE131_Incorrect_Calculation_Of_Buffer_Size__wchar_t_loop_03_good()
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
    CWE131_Incorrect_Calculation_Of_Buffer_Size__wchar_t_loop_03_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE131_Incorrect_Calculation_Of_Buffer_Size__wchar_t_loop_03_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
