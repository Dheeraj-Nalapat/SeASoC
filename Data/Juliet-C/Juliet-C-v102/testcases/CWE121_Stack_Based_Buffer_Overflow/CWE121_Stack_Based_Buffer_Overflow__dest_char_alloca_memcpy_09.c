/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE121_Stack_Based_Buffer_Overflow__dest_char_alloca_memcpy_09.c
Label Definition File: CWE121_Stack_Based_Buffer_Overflow__dest.string.label.xml
Template File: sources-sink-09.tmpl.c
*/
/*
 * @description
 * CWE: 121 Stack Based Buffer Overflow
 * BadSource:  Set data pointer to the bad buffer
 * GoodSource: Set data pointer to the good buffer
 * Sink: memcpy
 *    BadSink : Copy string to data using memcpy
 * Flow Variant: 09 Control flow: if(global_const_t) and if(global_const_f)
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#ifndef OMITBAD

void CWE121_Stack_Based_Buffer_Overflow__dest_char_alloca_memcpy_09_bad()
{
    char * data;
    char * data_badbuf = (char *)ALLOCA(50*sizeof(char));
    char * data_goodbuf = (char *)ALLOCA(100*sizeof(char));
    if(global_const_t)
    {
        /* FLAW: Set a pointer to a "small" buffer. This buffer will be used in the sinks as a destination
         * buffer in various memory copying functions using a "large" source buffer. */
        data = data_badbuf;
        data[0] = '\0'; /* null terminate */
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Set a pointer to a "large" buffer, thus avoiding buffer overflows in the sinks. */
        data = data_goodbuf;
        data[0] = '\0'; /* null terminate */
    }
    {
        char src[100];
        memset(src, 'C', 100-1); /* fill with 'C's */
        src[100-1] = '\0'; /* null terminate */
        /* POTENTIAL FLAW: Possible buffer overflow if the size of data is less than the length of src */
        memcpy(data, src, 100*sizeof(char));
        data[100-1] = '\0'; /* Ensure the destination buffer is null terminated */
        printLine(data);
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B1() - use goodsource and badsink by changing the global_const_t to global_const_f */
static void goodG2B1()
{
    char * data;
    char * data_badbuf = (char *)ALLOCA(50*sizeof(char));
    char * data_goodbuf = (char *)ALLOCA(100*sizeof(char));
    if(global_const_f)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FLAW: Set a pointer to a "small" buffer. This buffer will be used in the sinks as a destination
         * buffer in various memory copying functions using a "large" source buffer. */
        data = data_badbuf;
        data[0] = '\0'; /* null terminate */
    }
    else
    {
        /* FIX: Set a pointer to a "large" buffer, thus avoiding buffer overflows in the sinks. */
        data = data_goodbuf;
        data[0] = '\0'; /* null terminate */
    }
    {
        char src[100];
        memset(src, 'C', 100-1); /* fill with 'C's */
        src[100-1] = '\0'; /* null terminate */
        /* POTENTIAL FLAW: Possible buffer overflow if the size of data is less than the length of src */
        memcpy(data, src, 100*sizeof(char));
        data[100-1] = '\0'; /* Ensure the destination buffer is null terminated */
        printLine(data);
    }
}

/* goodG2B2() - use goodsource and badsink by reversing the blocks in the if statement */
static void goodG2B2()
{
    char * data;
    char * data_badbuf = (char *)ALLOCA(50*sizeof(char));
    char * data_goodbuf = (char *)ALLOCA(100*sizeof(char));
    if(global_const_t)
    {
        /* FIX: Set a pointer to a "large" buffer, thus avoiding buffer overflows in the sinks. */
        data = data_goodbuf;
        data[0] = '\0'; /* null terminate */
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FLAW: Set a pointer to a "small" buffer. This buffer will be used in the sinks as a destination
         * buffer in various memory copying functions using a "large" source buffer. */
        data = data_badbuf;
        data[0] = '\0'; /* null terminate */
    }
    {
        char src[100];
        memset(src, 'C', 100-1); /* fill with 'C's */
        src[100-1] = '\0'; /* null terminate */
        /* POTENTIAL FLAW: Possible buffer overflow if the size of data is less than the length of src */
        memcpy(data, src, 100*sizeof(char));
        data[100-1] = '\0'; /* Ensure the destination buffer is null terminated */
        printLine(data);
    }
}

void CWE121_Stack_Based_Buffer_Overflow__dest_char_alloca_memcpy_09_good()
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
    CWE121_Stack_Based_Buffer_Overflow__dest_char_alloca_memcpy_09_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE121_Stack_Based_Buffer_Overflow__dest_char_alloca_memcpy_09_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
