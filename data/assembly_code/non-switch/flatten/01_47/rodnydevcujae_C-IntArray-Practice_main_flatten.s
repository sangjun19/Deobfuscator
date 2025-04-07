	.file	"rodnydevcujae_C-IntArray-Practice_main_flatten.c"
	.text
	.globl	_TIG_IZ_Ibxm_argc
	.bss
	.align 4
	.type	_TIG_IZ_Ibxm_argc, @object
	.size	_TIG_IZ_Ibxm_argc, 4
_TIG_IZ_Ibxm_argc:
	.zero	4
	.globl	_TIG_IZ_Ibxm_envp
	.align 8
	.type	_TIG_IZ_Ibxm_envp, @object
	.size	_TIG_IZ_Ibxm_envp, 8
_TIG_IZ_Ibxm_envp:
	.zero	8
	.globl	_TIG_IZ_Ibxm_argv
	.align 8
	.type	_TIG_IZ_Ibxm_argv, @object
	.size	_TIG_IZ_Ibxm_argv, 8
_TIG_IZ_Ibxm_argv:
	.zero	8
	.text
	.globl	main
	.type	main, @function
main:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$880, %rsp
	movl	%edi, -852(%rbp)
	movq	%rsi, -864(%rbp)
	movq	%rdx, -872(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_Ibxm_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_Ibxm_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_Ibxm_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 109 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-Ibxm--0
# 0 "" 2
#NO_APP
	movl	-852(%rbp), %eax
	movl	%eax, _TIG_IZ_Ibxm_argc(%rip)
	movq	-864(%rbp), %rax
	movq	%rax, _TIG_IZ_Ibxm_argv(%rip)
	movq	-872(%rbp), %rax
	movq	%rax, _TIG_IZ_Ibxm_envp(%rip)
	nop
	movq	$5, -824(%rbp)
.L18:
	cmpq	$9, -824(%rbp)
	ja	.L21
	movq	-824(%rbp), %rax
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
	.long	.L13-.L8
	.long	.L21-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L21-.L8
	.long	.L10-.L8
	.long	.L21-.L8
	.long	.L21-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L9:
	movl	$5, -816(%rbp)
	movl	$-9, -812(%rbp)
	movl	$16, -808(%rbp)
	movl	$2, -804(%rbp)
	movl	$4, -836(%rbp)
	movq	$3, -824(%rbp)
	jmp	.L14
.L11:
	cmpl	$99, -836(%rbp)
	jbe	.L15
	movq	$2, -824(%rbp)
	jmp	.L14
.L15:
	movq	$9, -824(%rbp)
	jmp	.L14
.L7:
	movl	-836(%rbp), %eax
	movl	$0, -816(%rbp,%rax,4)
	addl	$1, -836(%rbp)
	movq	$3, -824(%rbp)
	jmp	.L14
.L10:
	movq	$8, -824(%rbp)
	jmp	.L14
.L13:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L19
	jmp	.L20
.L12:
	movl	$4, -832(%rbp)
	movl	$0, -828(%rbp)
	leaq	-416(%rbp), %rdx
	movl	-832(%rbp), %ecx
	leaq	-816(%rbp), %rax
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	filter
	movl	%eax, -828(%rbp)
	movl	-832(%rbp), %edx
	leaq	-816(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	printArray
	movl	-828(%rbp), %edx
	leaq	-416(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	printArray
	movq	$0, -824(%rbp)
	jmp	.L14
.L21:
	nop
.L14:
	jmp	.L18
.L20:
	call	__stack_chk_fail@PLT
.L19:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	main, .-main
	.section	.rodata
.LC0:
	.string	"]"
.LC1:
	.string	" %d "
	.text
	.globl	printArray
	.type	printArray, @function
printArray:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movl	%esi, -28(%rbp)
	movq	$0, -8(%rbp)
.L35:
	cmpq	$8, -8(%rbp)
	ja	.L36
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L25(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L25(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L25:
	.long	.L30-.L25
	.long	.L36-.L25
	.long	.L29-.L25
	.long	.L28-.L25
	.long	.L36-.L25
	.long	.L36-.L25
	.long	.L27-.L25
	.long	.L26-.L25
	.long	.L37-.L25
	.text
.L28:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$8, -8(%rbp)
	jmp	.L32
.L27:
	movl	$91, %edi
	call	putchar@PLT
	movl	$0, -12(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L32
.L30:
	movq	$6, -8(%rbp)
	jmp	.L32
.L26:
	movl	-12(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jge	.L33
	movq	$2, -8(%rbp)
	jmp	.L32
.L33:
	movq	$3, -8(%rbp)
	jmp	.L32
.L29:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -12(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L32
.L36:
	nop
.L32:
	jmp	.L35
.L37:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	printArray, .-printArray
	.globl	filter
	.type	filter, @function
filter:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -40(%rbp)
	movl	%esi, -44(%rbp)
	movq	%rdx, -56(%rbp)
	movq	$2, -8(%rbp)
.L54:
	cmpq	$9, -8(%rbp)
	ja	.L56
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L41(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L41(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L41:
	.long	.L47-.L41
	.long	.L56-.L41
	.long	.L46-.L41
	.long	.L56-.L41
	.long	.L45-.L41
	.long	.L44-.L41
	.long	.L43-.L41
	.long	.L56-.L41
	.long	.L42-.L41
	.long	.L40-.L41
	.text
.L45:
	movl	-20(%rbp), %eax
	movl	%eax, -12(%rbp)
	addl	$1, -20(%rbp)
	movl	-16(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	-12(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-56(%rbp), %rdx
	addq	%rcx, %rdx
	movl	(%rax), %eax
	movl	%eax, (%rdx)
	movq	$9, -8(%rbp)
	jmp	.L48
.L42:
	movl	$0, -20(%rbp)
	movl	$0, -16(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L48
.L40:
	addl	$1, -16(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L48
.L43:
	movl	-20(%rbp), %eax
	jmp	.L55
.L44:
	movl	-16(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	andl	$1, %eax
	testl	%eax, %eax
	jne	.L50
	movq	$4, -8(%rbp)
	jmp	.L48
.L50:
	movq	$9, -8(%rbp)
	jmp	.L48
.L47:
	movl	-16(%rbp), %eax
	cmpl	-44(%rbp), %eax
	jge	.L52
	movq	$5, -8(%rbp)
	jmp	.L48
.L52:
	movq	$6, -8(%rbp)
	jmp	.L48
.L46:
	movq	$8, -8(%rbp)
	jmp	.L48
.L56:
	nop
.L48:
	jmp	.L54
.L55:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	filter, .-filter
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
