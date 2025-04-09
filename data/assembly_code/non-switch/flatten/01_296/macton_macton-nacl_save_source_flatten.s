	.file	"macton_macton-nacl_save_source_flatten.c"
	.text
	.globl	_TIG_IZ_GWEN_argv
	.bss
	.align 8
	.type	_TIG_IZ_GWEN_argv, @object
	.size	_TIG_IZ_GWEN_argv, 8
_TIG_IZ_GWEN_argv:
	.zero	8
	.globl	_TIG_IZ_GWEN_envp
	.align 8
	.type	_TIG_IZ_GWEN_envp, @object
	.size	_TIG_IZ_GWEN_envp, 8
_TIG_IZ_GWEN_envp:
	.zero	8
	.globl	_TIG_IZ_GWEN_argc
	.align 4
	.type	_TIG_IZ_GWEN_argc, @object
	.size	_TIG_IZ_GWEN_argc, 4
_TIG_IZ_GWEN_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"Can't read the source"
	.align 8
.LC1:
	.string	"{ \"Ready\":false, \"Message\": \"%s\" }\r\n\r\n"
.LC2:
	.string	"Can't save the source"
.LC3:
	.string	"w"
.LC4:
	.string	"../hello.c"
.LC5:
	.string	"Source saved"
	.align 8
.LC6:
	.string	"{ \"Ready\":true, \"Message\": \"%s\", \"Id\": %d }\r\n\r\n"
	.align 8
.LC7:
	.string	"Content-type: application/json\r\n\r"
.LC8:
	.string	"CONTENT_LENGTH"
.LC9:
	.string	"%ld"
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
	leaq	-2097152(%rsp), %r11
.LPSRL0:
	subq	$4096, %rsp
	orq	$0, (%rsp)
	cmpq	%r11, %rsp
	jne	.LPSRL0
	subq	$112, %rsp
	movl	%edi, -2097236(%rbp)
	movq	%rsi, -2097248(%rbp)
	movq	%rdx, -2097256(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_GWEN_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_GWEN_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_GWEN_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 92 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-GWEN--0
# 0 "" 2
#NO_APP
	movl	-2097236(%rbp), %eax
	movl	%eax, _TIG_IZ_GWEN_argc(%rip)
	movq	-2097248(%rbp), %rax
	movq	%rax, _TIG_IZ_GWEN_argv(%rip)
	movq	-2097256(%rbp), %rax
	movq	%rax, _TIG_IZ_GWEN_envp(%rip)
	nop
	movq	$0, -2097200(%rbp)
.L33:
	cmpq	$19, -2097200(%rbp)
	ja	.L36
	movq	-2097200(%rbp), %rax
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
	.long	.L22-.L8
	.long	.L21-.L8
	.long	.L36-.L8
	.long	.L20-.L8
	.long	.L36-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L36-.L8
	.long	.L13-.L8
	.long	.L36-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L36-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L9:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L34
	jmp	.L35
.L12:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rdx
	leaq	-2097184(%rbp), %rax
	movl	$1048576, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	movq	$18, -2097200(%rbp)
	jmp	.L24
.L11:
	movq	-2097224(%rbp), %rax
	cmpq	$1048576, %rax
	jle	.L25
	movq	$3, -2097200(%rbp)
	jmp	.L24
.L25:
	movq	$9, -2097200(%rbp)
	jmp	.L24
.L13:
	cmpq	$0, -2097208(%rbp)
	jne	.L27
	movq	$1, -2097200(%rbp)
	jmp	.L24
.L27:
	movq	$8, -2097200(%rbp)
	jmp	.L24
.L16:
	movq	-2097208(%rbp), %rdx
	leaq	-1048592(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fputs@PLT
	movq	$17, -2097200(%rbp)
	jmp	.L24
.L21:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$17, -2097200(%rbp)
	jmp	.L24
.L20:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rdx
	leaq	-2097184(%rbp), %rax
	movl	$1048576, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	movq	$18, -2097200(%rbp)
	jmp	.L24
.L15:
	movq	stdin(%rip), %rdx
	movq	-2097224(%rbp), %rax
	addl	$1, %eax
	movl	%eax, %ecx
	leaq	-2097184(%rbp), %rax
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	movq	-2097224(%rbp), %rax
	movq	%rax, %rdx
	leaq	-2097184(%rbp), %rax
	leaq	(%rax,%rdx), %rcx
	leaq	-2097184(%rbp), %rax
	addq	$4, %rax
	leaq	-1048592(%rbp), %rdx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	unencode
	leaq	.LC3(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -2097208(%rbp)
	movq	$12, -2097200(%rbp)
	jmp	.L24
.L7:
	cmpl	$1, -2097228(%rbp)
	je	.L29
	movq	$14, -2097200(%rbp)
	jmp	.L24
.L29:
	movq	$15, -2097200(%rbp)
	jmp	.L24
.L10:
	movq	-2097208(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movl	-2097232(%rbp), %eax
	movl	%eax, %edx
	leaq	.LC5(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$18, -2097200(%rbp)
	jmp	.L24
.L18:
	movl	$0, %edi
	call	time@PLT
	movq	%rax, -2097192(%rbp)
	movq	-2097192(%rbp), %rax
	movl	%eax, -2097232(%rbp)
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	call	getenv@PLT
	movq	%rax, -2097216(%rbp)
	movq	$10, -2097200(%rbp)
	jmp	.L24
.L19:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rdx
	leaq	-2097184(%rbp), %rax
	movl	$1048576, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	movq	$18, -2097200(%rbp)
	jmp	.L24
.L14:
	cmpq	$0, -2097216(%rbp)
	jne	.L31
	movq	$5, -2097200(%rbp)
	jmp	.L24
.L31:
	movq	$7, -2097200(%rbp)
	jmp	.L24
.L22:
	movq	$6, -2097200(%rbp)
	jmp	.L24
.L17:
	leaq	-2097224(%rbp), %rdx
	movq	-2097216(%rbp), %rax
	leaq	.LC9(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_sscanf@PLT
	movl	%eax, -2097228(%rbp)
	movq	$19, -2097200(%rbp)
	jmp	.L24
.L36:
	nop
.L24:
	jmp	.L33
.L35:
	call	__stack_chk_fail@PLT
.L34:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	main, .-main
	.section	.rodata
.LC10:
	.string	"%2x"
	.text
	.globl	unencode
	.type	unencode, @function
unencode:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movq	%rdi, -40(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$12, -16(%rbp)
.L63:
	cmpq	$15, -16(%rbp)
	ja	.L66
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L40(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L40(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L40:
	.long	.L52-.L40
	.long	.L51-.L40
	.long	.L66-.L40
	.long	.L50-.L40
	.long	.L49-.L40
	.long	.L66-.L40
	.long	.L48-.L40
	.long	.L47-.L40
	.long	.L67-.L40
	.long	.L45-.L40
	.long	.L44-.L40
	.long	.L43-.L40
	.long	.L42-.L40
	.long	.L66-.L40
	.long	.L41-.L40
	.long	.L39-.L40
	.text
.L49:
	cmpl	$1, -20(%rbp)
	je	.L53
	movq	$14, -16(%rbp)
	jmp	.L55
.L53:
	movq	$0, -16(%rbp)
	jmp	.L55
.L41:
	movl	$63, -24(%rbp)
	movq	$0, -16(%rbp)
	jmp	.L55
.L39:
	addq	$1, -40(%rbp)
	addq	$1, -56(%rbp)
	movq	$10, -16(%rbp)
	jmp	.L55
.L42:
	movq	$10, -16(%rbp)
	jmp	.L55
.L51:
	movq	-40(%rbp), %rax
	leaq	1(%rax), %rcx
	leaq	-24(%rbp), %rax
	movq	%rax, %rdx
	leaq	.LC10(%rip), %rax
	movq	%rax, %rsi
	movq	%rcx, %rdi
	movl	$0, %eax
	call	__isoc99_sscanf@PLT
	movl	%eax, -20(%rbp)
	movq	$4, -16(%rbp)
	jmp	.L55
.L50:
	movq	-40(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$37, %al
	jne	.L57
	movq	$1, -16(%rbp)
	jmp	.L55
.L57:
	movq	$11, -16(%rbp)
	jmp	.L55
.L43:
	movq	-40(%rbp), %rax
	movzbl	(%rax), %edx
	movq	-56(%rbp), %rax
	movb	%dl, (%rax)
	movq	$15, -16(%rbp)
	jmp	.L55
.L45:
	movq	-56(%rbp), %rax
	movb	$10, (%rax)
	addq	$1, -56(%rbp)
	movq	-56(%rbp), %rax
	movb	$0, (%rax)
	movq	$8, -16(%rbp)
	jmp	.L55
.L48:
	movq	-56(%rbp), %rax
	movb	$32, (%rax)
	movq	$15, -16(%rbp)
	jmp	.L55
.L44:
	movq	-40(%rbp), %rdx
	movq	-48(%rbp), %rax
	cmpq	%rax, %rdx
	je	.L59
	movq	$7, -16(%rbp)
	jmp	.L55
.L59:
	movq	$9, -16(%rbp)
	jmp	.L55
.L52:
	movl	-24(%rbp), %eax
	movl	%eax, %edx
	movq	-56(%rbp), %rax
	movb	%dl, (%rax)
	addq	$2, -40(%rbp)
	movq	$15, -16(%rbp)
	jmp	.L55
.L47:
	movq	-40(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$43, %al
	jne	.L61
	movq	$6, -16(%rbp)
	jmp	.L55
.L61:
	movq	$3, -16(%rbp)
	jmp	.L55
.L66:
	nop
.L55:
	jmp	.L63
.L67:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L65
	call	__stack_chk_fail@PLT
.L65:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	unencode, .-unencode
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
