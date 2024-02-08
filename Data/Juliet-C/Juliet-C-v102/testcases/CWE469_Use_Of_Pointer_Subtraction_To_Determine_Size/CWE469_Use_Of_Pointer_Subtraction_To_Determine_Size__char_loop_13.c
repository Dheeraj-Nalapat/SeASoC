/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE469_Use_Of_Pointer_Subtraction_To_Determine_Size__char_loop_13.c
Label Definition File: CWE469_Use_Of_Pointer_Subtraction_To_Determine_Size.label.xml
Template File: point-flaw-13.tmpl.c
*/
/*
 * @description
 * CWE: 469 Use Of Pointer Subtraction To Determine Size
 * Sinks: loop
 *    GoodSink: using correct pointer subtraction to determine length in loop
 *    BadSink : Beginning and end pointers are switched leading to an incorrect length in the loop
 * Flow Variant: 13 Control flow: if(global_const_five==5) and if(global_const_five!=5)
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#define SRC_STR "abc/opqrstu"

#ifndef OMITBAD

void CWE469_Use_Of_Pointer_Subtraction_To_Determine_Size__char_loop_13_bad()
{
    if(global_const_five==5)
    {
        {
            char * data = SRC_STR;
            char dst[100] = "";
            char * end = strchr(data, '/');
            ptrdiff_t i;
            /* FLAW: switched the beginning and end pointers around */
            for (i = 0; i < (data - end); i++)
            {
                dst[i] = data[i];
            }
            printLine(dst);
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            char * data = SRC_STR;
            char dst[100] = "";
            char * end = strchr(data, '/');
            ptrdiff_t i;
            /* FIX: use proper order in pointer subtraction */
            for (i = 0; i < (end - data); i++)
            {
                dst[i] = data[i];
            }
            printLine(dst);
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good1() uses if(global_const_five!=5) instead of if(global_const_five==5) */
static void good1()
{
    if(global_const_five!=5)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            char * data = SRC_STR;
            char dst[100] = "";
            char * end = strchr(data, '/');
            ptrdiff_t i;
            /* FLAW: switched the beginning and end pointers around */
            for (i = 0; i < (data - end); i++)
            {
                dst[i] = data[i];
            }
            printLine(dst);
        }
    }
    else
    {
        {
            char * data = SRC_STR;
            char dst[100] = "";
            char * end = strchr(data, '/');
            ptrdiff_t i;
            /* FIX: use proper order in pointer subtraction */
            for (i = 0; i < (end - data); i++)
            {
                dst[i] = data[i];
            }
            printLine(dst);
        }
    }
}

/* good2() reverses the bodies in the if statement */
static void good2()
{
    if(global_const_five==5)
    {
        {
            char * data = SRC_STR;
            char dst[100] = "";
            char * end = strchr(data, '/');
            ptrdiff_t i;
            /* FIX: use proper order in pointer subtraction */
            for (i = 0; i < (end - data); i++)
            {
                dst[i] = data[i];
            }
            printLine(dst);
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            char * data = SRC_STR;
            char dst[100] = "";
            char * end = strchr(data, '/');
            ptrdiff_t i;
            /* FLAW: switched the beginning and end pointers around */
            for (i = 0; i < (data - end); i++)
            {
                dst[i] = data[i];
            }
            printLine(dst);
        }
    }
}

void CWE469_Use_Of_Pointer_Subtraction_To_Determine_Size__char_loop_13_good()
{
    good1();
    good2();
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
    CWE469_Use_Of_Pointer_Subtraction_To_Determine_Size__char_loop_13_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE469_Use_Of_Pointer_Subtraction_To_Determine_Size__char_loop_13_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
