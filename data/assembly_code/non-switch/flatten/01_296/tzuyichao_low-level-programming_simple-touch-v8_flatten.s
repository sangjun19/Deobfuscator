	.file	"tzuyichao_low-level-programming_simple-touch-v8_flatten.c"
	.text
	.globl	_TIG_IZ_Al8P_argv
	.bss
	.align 8
	.type	_TIG_IZ_Al8P_argv, @object
	.size	_TIG_IZ_Al8P_argv, 8
_TIG_IZ_Al8P_argv:
	.zero	8
	.globl	_TIG_IZ_Al8P_envp
	.align 8
	.type	_TIG_IZ_Al8P_envp, @object
	.size	_TIG_IZ_Al8P_envp, 8
_TIG_IZ_Al8P_envp:
	.zero	8
	.globl	_TIG_IZ_Al8P_argc
	.align 4
	.type	_TIG_IZ_Al8P_argc, @object
	.size	_TIG_IZ_Al8P_argc, 4
_TIG_IZ_Al8P_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"Can't create file"
.LC1:
	.string	"Can't update timestamp"
.LC2:
	.string	"You must supply a filename"
	.text
	.globl	main
	.type	main, @function
main:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$192, %rsp
	movl	%edi, -164(%rbp)
	movq	%rsi, -176(%rbp)
	movq	%rdx, -184(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_Al8P_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_Al8P_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_Al8P_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 131 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-Al8P--0
# 0 "" 2
#NO_APP
	movl	-164(%rbp), %eax
	movl	%eax, _TIG_IZ_Al8P_argc(%rip)
	movq	-176(%rbp), %rax
	movq	%rax, _TIG_IZ_Al8P_argv(%rip)
	movq	-184(%rbp), %rax
	movq	%rax, _TIG_IZ_Al8P_envp(%rip)
	nop
	movq	$23, -120(%rbp)
.L38:
	cmpq	$23, -120(%rbp)
	ja	.L41
	movq	-120(%rbp), %rax
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
	.long	.L25-.L8
	.long	.L41-.L8
	.long	.L24-.L8
	.long	.L23-.L8
	.long	.L22-.L8
	.long	.L21-.L8
	.long	.L41-.L8
	.long	.L20-.L8
	.long	.L41-.L8
	.long	.L19-.L8
	.long	.L41-.L8
	.long	.L41-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L41-.L8
	.long	.L7-.L8
	.text
.L12:
	call	__errno_location@PLT
	movq	%rax, -128(%rbp)
	movq	$5, -120(%rbp)
	jmp	.L26
.L22:
	leaq	-112(%rbp), %rax
	movl	$420, %esi
	movq	%rax, %rdi
	call	creat@PLT
	movl	%eax, -152(%rbp)
	movq	$13, -120(%rbp)
	jmp	.L26
.L16:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	call	__errno_location@PLT
	movq	%rax, -144(%rbp)
	movq	$2, -120(%rbp)
	jmp	.L26
.L15:
	movq	-176(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rcx
	leaq	-112(%rbp), %rax
	movl	$99, %edx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	strncat@PLT
	leaq	-112(%rbp), %rax
	movl	$0, %esi
	movq	%rax, %rdi
	call	utime@PLT
	movl	%eax, -148(%rbp)
	movq	$16, -120(%rbp)
	jmp	.L26
.L18:
	movl	-156(%rbp), %eax
	movb	$0, -112(%rbp,%rax)
	addl	$1, -156(%rbp)
	movq	$0, -120(%rbp)
	jmp	.L26
.L7:
	movq	$17, -120(%rbp)
	jmp	.L26
.L23:
	cmpl	$2, -164(%rbp)
	je	.L27
	movq	$7, -120(%rbp)
	jmp	.L26
.L27:
	movq	$15, -120(%rbp)
	jmp	.L26
.L14:
	cmpl	$-1, -148(%rbp)
	jne	.L29
	movq	$18, -120(%rbp)
	jmp	.L26
.L29:
	movq	$21, -120(%rbp)
	jmp	.L26
.L9:
	movl	$0, %eax
	jmp	.L39
.L19:
	movl	$1, %eax
	jmp	.L39
.L17:
	cmpl	$-1, -152(%rbp)
	jne	.L32
	movq	$14, -120(%rbp)
	jmp	.L26
.L32:
	movq	$21, -120(%rbp)
	jmp	.L26
.L11:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	call	__errno_location@PLT
	movq	%rax, -136(%rbp)
	movq	$20, -120(%rbp)
	jmp	.L26
.L13:
	movb	$0, -112(%rbp)
	movl	$1, -156(%rbp)
	movq	$0, -120(%rbp)
	jmp	.L26
.L21:
	movq	-128(%rbp), %rax
	movl	(%rax), %eax
	cmpl	$2, %eax
	jne	.L34
	movq	$4, -120(%rbp)
	jmp	.L26
.L34:
	movq	$19, -120(%rbp)
	jmp	.L26
.L25:
	cmpl	$99, -156(%rbp)
	jbe	.L36
	movq	$3, -120(%rbp)
	jmp	.L26
.L36:
	movq	$12, -120(%rbp)
	jmp	.L26
.L20:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$26, %edx
	movl	$1, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	$9, -120(%rbp)
	jmp	.L26
.L24:
	movq	-144(%rbp), %rax
	movl	(%rax), %eax
	jmp	.L39
.L10:
	movq	-136(%rbp), %rax
	movl	(%rax), %eax
	jmp	.L39
.L41:
	nop
.L26:
	jmp	.L38
.L39:
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L40
	call	__stack_chk_fail@PLT
.L40:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
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
