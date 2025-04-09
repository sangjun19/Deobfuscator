	.file	"DerrickMarcus_2023thuee_da_oj_2_flatten.c"
	.text
	.globl	_TIG_IZ_O2Ut_argc
	.bss
	.align 4
	.type	_TIG_IZ_O2Ut_argc, @object
	.size	_TIG_IZ_O2Ut_argc, 4
_TIG_IZ_O2Ut_argc:
	.zero	4
	.globl	_TIG_IZ_O2Ut_argv
	.align 8
	.type	_TIG_IZ_O2Ut_argv, @object
	.size	_TIG_IZ_O2Ut_argv, 8
_TIG_IZ_O2Ut_argv:
	.zero	8
	.globl	_TIG_IZ_O2Ut_envp
	.align 8
	.type	_TIG_IZ_O2Ut_envp, @object
	.size	_TIG_IZ_O2Ut_envp, 8
_TIG_IZ_O2Ut_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"%d"
.LC1:
	.string	"%d\n"
.LC2:
	.string	"%d%d"
	.text
	.globl	main
	.type	main, @function
main:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	leaq	-798720(%rsp), %r11
.LPSRL0:
	subq	$4096, %rsp
	orq	$0, (%rsp)
	cmpq	%r11, %rsp
	jne	.LPSRL0
	subq	$1408, %rsp
	movl	%edi, -800100(%rbp)
	movq	%rsi, -800112(%rbp)
	movq	%rdx, -800120(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_O2Ut_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_O2Ut_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_O2Ut_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 134 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-O2Ut--0
# 0 "" 2
#NO_APP
	movl	-800100(%rbp), %eax
	movl	%eax, _TIG_IZ_O2Ut_argc(%rip)
	movq	-800112(%rbp), %rax
	movq	%rax, _TIG_IZ_O2Ut_argv(%rip)
	movq	-800120(%rbp), %rax
	movq	%rax, _TIG_IZ_O2Ut_envp(%rip)
	nop
	movq	$35, -800040(%rbp)
.L48:
	movq	-800040(%rbp), %rax
	subq	$3, %rax
	cmpq	$40, %rax
	ja	.L51
	leaq	0(,%rax,4), %rdx
	leaq	.L8(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L8(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L8:
	.long	.L30-.L8
	.long	.L51-.L8
	.long	.L29-.L8
	.long	.L51-.L8
	.long	.L28-.L8
	.long	.L27-.L8
	.long	.L26-.L8
	.long	.L51-.L8
	.long	.L51-.L8
	.long	.L25-.L8
	.long	.L51-.L8
	.long	.L24-.L8
	.long	.L51-.L8
	.long	.L51-.L8
	.long	.L51-.L8
	.long	.L51-.L8
	.long	.L23-.L8
	.long	.L22-.L8
	.long	.L21-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L51-.L8
	.long	.L51-.L8
	.long	.L51-.L8
	.long	.L51-.L8
	.long	.L16-.L8
	.long	.L51-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L51-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L51-.L8
	.long	.L51-.L8
	.long	.L9-.L8
	.long	.L51-.L8
	.long	.L7-.L8
	.text
.L17:
	movl	$-1, -800044(%rbp)
	movl	-800060(%rbp), %eax
	cltq
	movl	-800044(%rbp), %edx
	movl	%edx, -400016(%rbp,%rax,4)
	movl	-800060(%rbp), %eax
	cltq
	movl	-800044(%rbp), %edx
	movl	%edx, -800032(%rbp,%rax,4)
	addl	$1, -800060(%rbp)
	movq	$43, -800040(%rbp)
	jmp	.L31
.L16:
	cmpl	$100000, -800068(%rbp)
	jbe	.L32
	movq	$24, -800040(%rbp)
	jmp	.L31
.L32:
	movq	$12, -800040(%rbp)
	jmp	.L31
.L24:
	movl	-800084(%rbp), %eax
	cmpl	%eax, -800056(%rbp)
	jge	.L34
	movq	$33, -800040(%rbp)
	jmp	.L31
.L34:
	movq	$22, -800040(%rbp)
	jmp	.L31
.L25:
	movl	-800068(%rbp), %eax
	movl	$0, -800032(%rbp,%rax,4)
	addl	$1, -800068(%rbp)
	movq	$30, -800040(%rbp)
	jmp	.L31
.L27:
	cmpl	$100000, -800064(%rbp)
	jbe	.L36
	movq	$7, -800040(%rbp)
	jmp	.L31
.L36:
	movq	$9, -800040(%rbp)
	jmp	.L31
.L19:
	movq	$21, -800040(%rbp)
	jmp	.L31
.L30:
	leaq	-800076(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-800076(%rbp), %eax
	cltq
	movl	-800032(%rbp,%rax,4), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$21, -800040(%rbp)
	jmp	.L31
.L18:
	movl	$1, -400016(%rbp)
	movl	$0, -400012(%rbp)
	movl	$2, -800064(%rbp)
	movq	$8, -800040(%rbp)
	jmp	.L31
.L21:
	addl	$1, -800056(%rbp)
	movq	$14, -800040(%rbp)
	jmp	.L31
.L12:
	leaq	-800072(%rbp), %rdx
	leaq	-800076(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-800076(%rbp), %eax
	movl	-800072(%rbp), %ecx
	cltq
	movl	-800032(%rbp,%rax,4), %edx
	movslq	%ecx, %rax
	movl	%edx, -800032(%rbp,%rax,4)
	movl	-800076(%rbp), %eax
	cltq
	movl	-800032(%rbp,%rax,4), %eax
	movl	-800072(%rbp), %edx
	cltq
	movl	%edx, -400016(%rbp,%rax,4)
	movl	-800076(%rbp), %eax
	movl	-800072(%rbp), %edx
	cltq
	movl	%edx, -800032(%rbp,%rax,4)
	movl	-800072(%rbp), %eax
	movl	-800076(%rbp), %edx
	cltq
	movl	%edx, -400016(%rbp,%rax,4)
	movq	$21, -800040(%rbp)
	jmp	.L31
.L26:
	movl	-800064(%rbp), %eax
	movl	$0, -400016(%rbp,%rax,4)
	addl	$1, -800064(%rbp)
	movq	$8, -800040(%rbp)
	jmp	.L31
.L23:
	leaq	-800084(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$1, -800032(%rbp)
	movl	$0, -800028(%rbp)
	movl	$2, -800068(%rbp)
	movq	$30, -800040(%rbp)
	jmp	.L31
.L15:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L49
	jmp	.L50
.L10:
	movl	-800052(%rbp), %eax
	cltq
	movl	-800032(%rbp,%rax,4), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-800052(%rbp), %eax
	cltq
	movl	-800032(%rbp,%rax,4), %eax
	movl	%eax, -800052(%rbp)
	movq	$37, -800040(%rbp)
	jmp	.L31
.L20:
	movl	$0, -800052(%rbp)
	movq	$37, -800040(%rbp)
	jmp	.L31
.L29:
	movl	-800080(%rbp), %eax
	cmpl	$3, %eax
	je	.L39
	cmpl	$3, %eax
	jg	.L40
	cmpl	$1, %eax
	je	.L41
	cmpl	$2, %eax
	je	.L42
	jmp	.L40
.L39:
	movq	$41, -800040(%rbp)
	jmp	.L43
.L42:
	movq	$3, -800040(%rbp)
	jmp	.L43
.L41:
	movq	$36, -800040(%rbp)
	jmp	.L43
.L40:
	movq	$23, -800040(%rbp)
	nop
.L43:
	jmp	.L31
.L14:
	leaq	-800080(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$5, -800040(%rbp)
	jmp	.L31
.L11:
	movl	-800052(%rbp), %eax
	cltq
	movl	-800032(%rbp,%rax,4), %eax
	testl	%eax, %eax
	je	.L44
	movq	$38, -800040(%rbp)
	jmp	.L31
.L44:
	movq	$32, -800040(%rbp)
	jmp	.L31
.L9:
	leaq	-800076(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-800076(%rbp), %edx
	movl	-800076(%rbp), %eax
	cltq
	movl	-800032(%rbp,%rax,4), %ecx
	movslq	%edx, %rax
	movl	-400016(%rbp,%rax,4), %edx
	movslq	%ecx, %rax
	movl	%edx, -400016(%rbp,%rax,4)
	movl	-800076(%rbp), %edx
	movl	-800076(%rbp), %eax
	cltq
	movl	-400016(%rbp,%rax,4), %ecx
	movslq	%edx, %rax
	movl	-800032(%rbp,%rax,4), %edx
	movslq	%ecx, %rax
	movl	%edx, -800032(%rbp,%rax,4)
	movl	$-1, -800048(%rbp)
	movl	-800076(%rbp), %eax
	cltq
	movl	-800048(%rbp), %edx
	movl	%edx, -400016(%rbp,%rax,4)
	movl	-800076(%rbp), %eax
	cltq
	movl	-800048(%rbp), %edx
	movl	%edx, -800032(%rbp,%rax,4)
	movq	$21, -800040(%rbp)
	jmp	.L31
.L28:
	movl	$2, -800060(%rbp)
	movq	$43, -800040(%rbp)
	jmp	.L31
.L13:
	movq	$19, -800040(%rbp)
	jmp	.L31
.L7:
	cmpl	$100000, -800060(%rbp)
	jg	.L46
	movq	$25, -800040(%rbp)
	jmp	.L31
.L46:
	movq	$20, -800040(%rbp)
	jmp	.L31
.L22:
	movl	$0, -800056(%rbp)
	movq	$14, -800040(%rbp)
	jmp	.L31
.L51:
	nop
.L31:
	jmp	.L48
.L50:
	call	__stack_chk_fail@PLT
.L49:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	main, .-main
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
