/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE469_Use_Of_Pointer_Subtraction_To_Determine_Size__char_memmove_07.c
Label Definition File: CWE469_Use_Of_Pointer_Subtraction_To_Determine_Size.label.xml
Template File: point-flaw-07.tmpl.c
*/
/*
 * @description
 * CWE: 469 Use Of Pointer Subtraction To Determine Size
 * Sinks: memmove
 *    GoodSink: using correct pointer subtraction to determine length in memmove()
 *    BadSink : Beginning and end pointers are switched leading to an incorrect length in memmove()
 * Flow Variant: 07 Control flow: if(static_five==5) and if(static_five!=5)
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#define SRC_STR "abc/opqrstu"

/* The variable below is not declared "const", but is never assigned
   any other value so a tool should be able to identify that reads of
   this will always give its initialized value. */
static int static_five = 5;

#ifndef OMITBAD

void CWE469_Use_Of_Pointer_Subtraction_To_Determine_Size__char_memmove_07_bad()
{
    if(static_five==5)
    {
        {
            char * data = SRC_STR;
            char dst[100] = "";
            char * end = strchr(data, '/');
            /* FLAW: switched the beginning and end pointers around */
            memmove(dst, data, sizeof(char)*(data - end));
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
            /* FIX: use proper order in pointer subtraction */
            memmove(dst, data, sizeof(char)*(end - data));
            printLine(dst);
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good1() uses if(static_five!=5) instead of if(static_five==5) */
static void good1()
{
    if(static_five!=5)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            char * data = SRC_STR;
            char dst[100] = "";
            char * end = strchr(data, '/');
            /* FLAW: switched the beginning and end pointers around */
            memmove(dst, data, sizeof(char)*(data - end));
            printLine(dst);
        }
    }
    else
    {
        {
            char * data = SRC_STR;
            char dst[100] = "";
            char * end = strchr(data, '/');
            /* FIX: use proper order in pointer subtraction */
            memmove(dst, data, sizeof(char)*(end - data));
            printLine(dst);
        }
    }
}

/* good2() reverses the bodies in the if statement */
static void good2()
{
    if(static_five==5)
    {
        {
            char * data = SRC_STR;
            char dst[100] = "";
            char * end = strchr(data, '/');
            /* FIX: use proper order in pointer subtraction */
            memmove(dst, data, sizeof(char)*(end - data));
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
            /* FLAW: switched the beginning and end pointers around */
            memmove(dst, data, sizeof(char)*(data - end));
            printLine(dst);
        }
    }
}

void CWE469_Use_Of_Pointer_Subtraction_To_Determine_Size__char_memmove_07_good()
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
    CWE469_Use_Of_Pointer_Subtraction_To_Determine_Size__char_memmove_07_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE469_Use_Of_Pointer_Subtraction_To_Determine_Size__char_memmove_07_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
